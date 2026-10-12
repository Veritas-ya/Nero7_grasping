// Ideal position servos, belt motion, and geometric grasp constraint.
// Ground truth is diagnostic only; perception/control use rendered RGB-D.
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <gazebo_msgs/msg/model_states.hpp>
#include <array>
#include <atomic>
#include <cmath>
#include <mutex>
#include <sstream>

namespace nero7_sim {
class Scene : public gazebo::WorldPlugin {
 public:
  void Load(gazebo::physics::WorldPtr world, sdf::ElementPtr sdf) override {
    world_ = world; node_ = gazebo_ros::Node::Get(sdf);
    std::istringstream initial(sdf->Get<std::string>("initial_joints"));
    for (size_t i=0; i<7; ++i) initial >> target_[i];
    target_[7] = .07;
    tcp_ = sdf->Get<ignition::math::Vector3d>("tcp");
    auto direction = sdf->Get<ignition::math::Vector2d>("direction");
    velocity_.Set(direction.X(), direction.Y(), 0);
    velocity_.Normalize(); velocity_ *= sdf->Get<double>("belt_speed");
    auto gate = sdf->Get<ignition::math::Vector2d>("gate");
    ignition::math::Vector3d along(direction.X(), direction.Y(), 0); along.Normalize();
    start_.Set(gate.X(), gate.Y(), 0);
    start_ -= along*sdf->Get<double>("spawn_distance");
    size_ = sdf->Get<ignition::math::Vector3d>("object_size");
    belt_z_ = sdf->Get<double>("belt_height"); start_.Z(belt_z_+size_.Z()/2+.001);
    yaw_ = sdf->Get<double>("object_yaw");
    states_ = node_->create_publisher<sensor_msgs::msg::JointState>("/sim/nero7/gazebo/joint_states", 10);
    flange_states_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>("/sim/nero7/gazebo/flange_pose", 10);
    truth_ = node_->create_publisher<gazebo_msgs::msg::ModelStates>("/sim/scene/ground_truth", 10);
    events_ = node_->create_publisher<std_msgs::msg::String>("/sim/scene/grasp_event", 10);
    commands_ = node_->create_subscription<std_msgs::msg::Float64MultiArray>(
      "/sim/nero7/gazebo/joint_commands", 10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size()!=8) return;
        for (double value : msg->data) if (!std::isfinite(value)) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::copy(msg->data.begin(), msg->data.end(), target_.begin());
        target_[7] = std::max(0., std::min(.07, target_[7]));
      });
    spawn_ = node_->create_service<std_srvs::srv::Trigger>("/sim/belt/spawn",
      [this](std_srvs::srv::Trigger::Request::SharedPtr, std_srvs::srv::Trigger::Response::SharedPtr response) {
        spawn_requested_ = true; response->success = true;
        response->message = "One example block queued at the upstream entry";
      });
    running_ = node_->create_service<std_srvs::srv::SetBool>("/sim/belt/set_running",
      [this](std_srvs::srv::SetBool::Request::SharedPtr request, std_srvs::srv::SetBool::Response::SharedPtr response) {
        belt_running_ = request->data; response->success = true;
      });
    connection_ = gazebo::event::Events::ConnectWorldUpdateBegin(std::bind(&Scene::Update, this, std::placeholders::_1));
  }

 private:
  void Event(const std::string &value) {
    std_msgs::msg::String msg; msg.data=value; events_->publish(msg);
    RCLCPP_INFO(node_->get_logger(), "%s", value.c_str());
  }
  void Release() {
    if (attachment_) { attachment_->Detach(); attachment_->Fini(); attachment_.reset(); Event("RELEASED"); }
  }
  void Update(const gazebo::common::UpdateInfo &info) {
    if (!robot_ || !block_) {
      robot_ = world_->ModelByName("nero7"); block_ = world_->ModelByName("example_block");
      if (!robot_ || !block_) return;
      for (int i=0;i<7;++i) joints_[i]=robot_->GetJoint("joint"+std::to_string(i+1));
      joints_[7]=robot_->GetJoint("left_finger_joint"); joints_[8]=robot_->GetJoint("right_finger_joint");
      flange_=robot_->GetLink("link7"); body_=block_->GetLink("body");
      for (auto &joint : joints_) if (!joint) { RCLCPP_FATAL(node_->get_logger(), "Required URDF joint missing"); connection_.reset(); return; }
      if (!flange_ || !body_) { connection_.reset(); return; }
      body_->SetGravityMode(false);
    }
    std::array<double,8> command;
    { std::lock_guard<std::mutex> lock(mutex_); command=target_; }
    for (size_t i=0;i<7;++i) { joints_[i]->SetPosition(0,command[i],true); joints_[i]->SetVelocity(0,0); }
    if (spawn_requested_.exchange(false)) {
      Release(); block_->SetWorldPose(ignition::math::Pose3d(start_, ignition::math::Quaterniond(0,0,yaw_)));
      block_->SetLinearVel(ignition::math::Vector3d::Zero); block_->SetAngularVel(ignition::math::Vector3d::Zero);
      body_->SetGravityMode(true); spawned_=true; Event("SPAWNED");
    }
    double actual_width=command[7];
    if (attachment_ && command[7]>held_width_+.006) Release();
    if (spawned_ && !attachment_) {
      const auto flange=flange_->WorldPose(), object=block_->WorldPose();
      const auto tip=flange.Pos()+flange.Rot().RotateVector(tcp_);
      auto axis=flange.Rot().RotateVector(ignition::math::Vector3d(0,0,1));
      axis.Z(0); axis.Normalize();
      const auto delta=tip-object.Pos();
      const auto local_tip=object.Rot().RotateVectorReverse(delta);
      const auto object_x=object.Rot().RotateVector(ignition::math::Vector3d::UnitX);
      const auto object_y=object.Rot().RotateVector(ignition::math::Vector3d::UnitY);
      const double width=std::abs(axis.Dot(object_x))*size_.X()+std::abs(axis.Dot(object_y))*size_.Y();
      // Ideal grasp-region proxy: tip inside the box footprint, jaw closure
      // through its projected width, and usable insertion height. This does
      // not model friction/contact forces or require an exact centre grasp.
      if (width<.068 && command[7]<=width
          && std::abs(local_tip.X())<size_.X()/2+.003
          && std::abs(local_tip.Y())<size_.Y()/2+.003
          && std::abs(delta.Dot(ignition::math::Vector3d(-axis.Y(),axis.X(),0)))<.03
          && std::abs(delta.Z())<.025) {
        held_width_=width; actual_width=width;
        attachment_=world_->Physics()->CreateJoint("fixed",robot_);
        attachment_->Load(flange_,body_,ignition::math::Pose3d::Zero); attachment_->Init();
        Event("ATTACHED_GEOMETRIC_CONTACT");
      } else if (object.Pos().Z()-size_.Z()/2< belt_z_+.006) {
        block_->SetLinearVel(belt_running_ ? velocity_ : ignition::math::Vector3d::Zero);
        block_->SetAngularVel(ignition::math::Vector3d::Zero);
      }
    }
    if (attachment_) actual_width=std::max(command[7],held_width_);
    for (size_t i=7;i<9;++i) { joints_[i]->SetPosition(0,actual_width/2+.003,true); joints_[i]->SetVelocity(0,0); }
    if ((info.simTime-last_publish_).Double()<.02) return;
    last_publish_=info.simTime;
    sensor_msgs::msg::JointState feedback;
    feedback.header.stamp.sec=info.simTime.sec; feedback.header.stamp.nanosec=info.simTime.nsec;
    for (size_t i=0;i<9;++i) {
      feedback.name.push_back(i<7 ? "joint"+std::to_string(i+1) : (i==7 ? "left_finger_joint":"right_finger_joint"));
      feedback.position.push_back(joints_[i]->Position(0));
    }
    states_->publish(feedback);
    geometry_msgs::msg::PoseStamped flange_feedback; flange_feedback.header=feedback.header;
    flange_feedback.header.frame_id="base_link"; const auto f=flange_->WorldPose();
    flange_feedback.pose.position.x=f.Pos().X(); flange_feedback.pose.position.y=f.Pos().Y(); flange_feedback.pose.position.z=f.Pos().Z();
    flange_feedback.pose.orientation.x=f.Rot().X(); flange_feedback.pose.orientation.y=f.Rot().Y();
    flange_feedback.pose.orientation.z=f.Rot().Z(); flange_feedback.pose.orientation.w=f.Rot().W();
    flange_states_->publish(flange_feedback);
    if (spawned_) {
      gazebo_msgs::msg::ModelStates diagnostic; diagnostic.name={"example_block"};
      geometry_msgs::msg::Pose pose; const auto p=block_->WorldPose();
      pose.position.x=p.Pos().X(); pose.position.y=p.Pos().Y(); pose.position.z=p.Pos().Z();
      pose.orientation.x=p.Rot().X(); pose.orientation.y=p.Rot().Y(); pose.orientation.z=p.Rot().Z(); pose.orientation.w=p.Rot().W();
      diagnostic.pose.push_back(pose); geometry_msgs::msg::Twist twist; const auto v=block_->WorldLinearVel();
      twist.linear.x=v.X(); twist.linear.y=v.Y(); twist.linear.z=v.Z(); diagnostic.twist.push_back(twist);
      truth_->publish(diagnostic);
    }
  }
  gazebo::physics::WorldPtr world_; gazebo::physics::ModelPtr robot_,block_;
  gazebo::physics::LinkPtr flange_,body_; gazebo::physics::JointPtr attachment_;
  std::array<gazebo::physics::JointPtr,9> joints_;
  std::array<double,8> target_{}; std::mutex mutex_;
  ignition::math::Vector3d tcp_,velocity_,start_,size_;
  double belt_z_=0,yaw_=0,held_width_=0; bool spawned_=false;
  std::atomic<bool> spawn_requested_{false},belt_running_{true};
  gazebo::common::Time last_publish_; gazebo::event::ConnectionPtr connection_;
  gazebo_ros::Node::SharedPtr node_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr states_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr flange_states_;
  rclcpp::Publisher<gazebo_msgs::msg::ModelStates>::SharedPtr truth_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr events_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr commands_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr spawn_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr running_;
};
GZ_REGISTER_WORLD_PLUGIN(Scene)
}  // namespace nero7_sim
