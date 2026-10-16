"""Workspace verification and local execution handshake; never sends arm poses."""
import argparse
import importlib.util
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--execute',action='store_true')
    mode.add_argument('--close-gate',action='store_true')
    mode.add_argument('--check-workspace',type=Path)
    parser.add_argument('--session-file',type=Path)
    parser.add_argument('--owner-pid',type=int)
    args=parser.parse_args()
    if args.check_workspace:
        src=args.check_workspace/'src/nero7_control/nero7_control/static_once.py'
        installed=Path(importlib.util.find_spec('nero7_control.static_once').origin)
        expected=args.check_workspace/'install/nero7_control/lib/python3.10/site-packages/nero7_control/static_once.py'
        if installed.resolve()!=expected.resolve() or src.read_bytes()!=installed.read_bytes():
            raise RuntimeError('加载的 static_once 与当前根目录源码不一致：'+str(installed))
        print('已核对当前加载文件：'+str(installed),flush=True)
        return
    if args.session_file is None:raise RuntimeError('Missing session file')
    pid=args.owner_pid if args.close_gate else int(args.session_file.read_text().strip())
    marker=args.session_file.with_name(args.session_file.name+'.gate-'+str(pid))
    if args.close_gate and not marker.exists():return
    if args.execute:os.kill(pid,0)
    import rclpy
    from rcl_interfaces.srv import SetParameters,GetParameters
    from rcl_interfaces.msg import Parameter,ParameterValue,ParameterType
    from std_msgs.msg import String
    from std_srvs.srv import SetBool
    rclpy.init();node=rclpy.create_node('local_static_execution_handshake')
    gate=node.create_client(SetBool,'/control_enable')
    owned=False
    def call(client,request):
        if not client.wait_for_service(timeout_sec=2.):raise RuntimeError('服务不可用：'+client.srv_name)
        future=client.call_async(request)
        rclpy.spin_until_future_complete(node,future,timeout_sec=2.)
        if not future.done():raise RuntimeError('服务超时：'+client.srv_name)
        return future.result()
    def close():
        response=call(gate,SetBool.Request(data=False))
        if not response.success:raise RuntimeError(response.message)
        marker.unlink(missing_ok=True)
        print('本轮驱动控制门已关闭。',flush=True)
    try:
        if args.close_gate:
            close();return
        last={}
        sub=node.create_subscription(String,'/grasp/state',
            lambda m:last.update(state=m.data,received=time.monotonic()),5)
        end=time.monotonic()+3
        while time.monotonic()<end:
            rclpy.spin_once(node,timeout_sec=.05)
            if last.get('state')=='STATIC_READY':break
        if last.get('state')!='STATIC_READY' or time.monotonic()-last.get('received',0)>.3:
            raise RuntimeError('当前未就绪：'+last.get('state','未找到抓取节点')+'；未开启执行。')
        if sum(name=='grasp_controller' for name,_ in node.get_node_names_and_namespaces())!=1:
            raise RuntimeError('抓取节点缺失或重复；未开启执行。')
        getter=node.create_client(GetParameters,'/grasp_controller/get_parameters')
        values=call(getter,GetParameters.Request(names=['mode','execution_enabled'])).values
        if (len(values)!=2 or values[0].string_value!='static_once'
                or values[1].type!=ParameterType.PARAMETER_BOOL or values[1].bool_value):
            raise RuntimeError('节点不是等待执行的 static_once；未开启执行。')
        marker.write_text(str(pid));owned=True
        response=call(gate,SetBool.Request(data=True))
        if not response.success:raise RuntimeError(response.message)
        # Recheck state after opening the gate, before granting execution.
        if last.get('state')!='STATIC_READY' or time.monotonic()-last['received']>.3:
            raise RuntimeError('开启控制门期间目标状态已变化；未开启执行。')
        setter=node.create_client(SetParameters,'/grasp_controller/set_parameters')
        parameter=Parameter(name='execution_enabled',value=ParameterValue(
            type=ParameterType.PARAMETER_BOOL,bool_value=True))
        results=call(setter,SetParameters.Request(parameters=[parameter])).results
        if len(results)!=1 or not results[0].successful:raise RuntimeError(str(results))
        owned=False
        print('控制门已打开，已开启本轮执行。结束或中止后启动脚本会关闭控制门。',flush=True)
    finally:
        if owned:close()
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
