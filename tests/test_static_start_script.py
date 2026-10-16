"""Execution-script handshake against ROS test services, never a robot driver."""
import os
from pathlib import Path
import subprocess
import sys
import threading
import pytest
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import SetBool
from rcl_interfaces.msg import SetParametersResult

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_controller():
    rclpy.init()
    node=Node('grasp_controller')
    node.declare_parameter('mode','static_once')
    node.declare_parameter('execution_enabled',False)
    state={'value':'STATIC_READY','reject':False,'events':[]}
    pub=node.create_publisher(String,'/grasp/state',5)
    timer=node.create_timer(.05,lambda:pub.publish(String(data=state['value'])))
    def gate(req,response):
        state['events'].append(('gate',req.data));response.success=True;return response
    service=node.create_service(SetBool,'/control_enable',gate)
    def parameters(params):
        state['events'].extend(('execute',p.value) for p in params if p.name=='execution_enabled')
        return SetParametersResult(successful=not state['reject'])
    node.add_on_set_parameters_callback(parameters)
    executor=SingleThreadedExecutor();executor.add_node(node)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    yield state
    executor.shutdown();thread.join();node.destroy_node();rclpy.shutdown()


def invoke(tmp_path,*args):
    session=tmp_path/'session.pid';session.write_text(str(os.getpid()))
    result=subprocess.run([sys.executable,str(ROOT/'scripts/static_once_session.py'),
        '--session-file',str(session),*args],capture_output=True,text=True,timeout=12)
    return result,session


def test_ready_opens_gate_before_execution_and_cleanup_closes_it(fake_controller,tmp_path):
    result,session=invoke(tmp_path,'--execute')
    assert result.returncode==0,result.stderr
    assert fake_controller['events']==[('gate',True),('execute',True)]
    closed,_=invoke(tmp_path,'--close-gate','--owner-pid',str(os.getpid()))
    assert closed.returncode==0,closed.stderr
    assert fake_controller['events'][-1]==('gate',False)


def test_not_ready_never_opens_gate_or_enables(fake_controller,tmp_path):
    fake_controller['value']='STATIC_INFER'
    result,_=invoke(tmp_path,'--execute')
    assert result.returncode!=0
    assert fake_controller['events']==[]


def test_parameter_rejection_closes_gate(fake_controller,tmp_path):
    fake_controller['reject']=True
    result,_=invoke(tmp_path,'--execute')
    assert result.returncode!=0
    assert fake_controller['events']==[('gate',True),('execute',True),('gate',False)]
