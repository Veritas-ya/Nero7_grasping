# A tracked block on the conveyor belt, expressed in base_link frame.
int32 id                        # Track id, unique within a run

builtin_interfaces/Time stamp   # Last accepted image observation; never refreshed by prediction
builtin_interfaces/Time motion_stamp # Reference time of the filtered/predicted motion state

geometry_msgs/Point centroid_base  # Filtered XY at motion_stamp; Z from last accepted observation (m)
geometry_msgs/Point observed_centroid_base # Raw accepted centroid at stamp; for image-time grasp matching
geometry_msgs/Vector3 velocity_base # Kalman XY velocity in base_link (m/s); Z=0
bool speed_valid                # Enough accepted observations, bounded covariance, fresh observation
float64[16] motion_covariance    # Row-major 4x4 covariance of [x,y,vx,vy]
float64[4] motion_process_noise  # Row-major 2x2 white-acceleration spectral density (m^2/s^3)
float64 speed                   # Signed projection of Kalman velocity on belt direction; 0 if !speed_valid
float64 s                       # Scalar position along belt direction: (centroid-p0).d_hat (m)

float64 height                  # Block height above belt plane (m)
float64 belt_z                  # Belt plane z in base_link at the block (m)
float64 width_px                # Blob width in color image pixels
float64 u_px                    # Centroid u in color image grid
float64 v_px                    # Centroid v in color image grid
float64 confidence              # Detection confidence in [0, 1]

uint8 status                    # 0=tracking (not confirmed), 1=confirmed detection, 2=lost
