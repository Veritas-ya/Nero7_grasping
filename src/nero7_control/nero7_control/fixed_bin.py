"""Fixed, taught placement poses and preflight; no detector or robot commands."""
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from .ready_pose import require_limits


def load_fixed_bin(path):
    raw=Path(path).read_bytes()
    data=yaml.safe_load(raw)
    if (not isinstance(data,dict) or data.get('schema_version')!=1
            or data.get('frame_id')!='base_link' or data.get('units')!='m_rad'):
        raise ValueError('Fixed bin requires schema 1, base_link and m_rad')
    if data.get('configured') is False:
        return None
    if data.get('configured') is not True or data.get('pose_type') not in ('joints','tcp'):
        raise ValueError('Fixed bin must explicitly select configured and joints/tcp pose_type')
    size=7 if data['pose_type']=='joints' else 6
    suffix='joints_rad' if size==7 else 'tcp_pose_m_rad'
    for name in ('above_'+suffix,'release_'+suffix):
        value=np.asarray(data.get(name,[]),float)
        if value.shape!=(size,) or not np.isfinite(value).all():
            raise ValueError('Fixed bin requires '+name)
    for key in ('via_joints_rad','return_path_joints_rad'):
        via=np.asarray(data.get(key,[]),float)
        if via.size and (via.ndim!=2 or via.shape[1]!=7 or not np.isfinite(via).all()):
            raise ValueError('Fixed bin '+key+' points must each contain seven radians')
    if 'initial_joints_rad' in data:
        initial=np.asarray(data['initial_joints_rad'],float)
        if initial.shape!=(7,) or not np.isfinite(initial).all():
            raise ValueError('Fixed bin initial_joints_rad must contain seven radians')
    data=deepcopy(data)
    data['profile_path']=str(Path(path).resolve())
    data['sha256']=hashlib.sha256(raw).hexdigest()
    return data


def fixed_bin_profile_current(profile):
    try:
        return hashlib.sha256(Path(profile['profile_path']).read_bytes()).hexdigest()==profile['sha256']
    except (OSError,KeyError,TypeError):
        return False


def _point(kin,q,tcp,workspace):
    q=require_limits(kin,q)
    T=kin.fk(q);flange=T[:3,3]
    if not (workspace['x_min']<=flange[0]<=workspace['x_max']
            and workspace['y_min']<=flange[1]<=workspace['y_max']
            and flange[2]>=workspace['z_min']):
        raise ValueError('Fixed bin path leaves flange workspace')
    rotation=Rotation.from_matrix(T[:3,:3])
    return flange+rotation.apply(tcp),rotation


def _joint_route(kin,previous,route,tcp,workspace,floor):
    minimum=float('inf')
    for goal in route:
        if np.max(np.abs(goal-previous))>2.3:
            raise ValueError('Bin transfer joint step exceeds 2.3 rad; teach intermediate via points')
        count=max(150,int(np.ceil(np.max(np.abs(goal-previous))/.02))+1)
        for alpha in np.linspace(0.,1.,count):
            tip,_=_point(kin,previous*(1-alpha)+goal*alpha,tcp,workspace)
            minimum=min(minimum,float(tip[2]))
            if tip[2]<floor-1e-9:
                raise ValueError('Bin transfer dips below the required carry height')
        previous=goal
    return minimum


def plan_fixed_bin(kin,lift_joints,tcp,profile,belt_z,workspace):
    if profile is None:
        raise ValueError('Fixed material bin has not been taught')
    tcp=np.asarray(tcp,float)
    if tcp.shape!=(3,) or not np.isfinite(tcp).all() or not np.isfinite(belt_z):
        raise ValueError('Invalid fixed bin TCP/belt geometry')
    previous=require_limits(kin,lift_joints).copy()
    lift_tip,lift_rotation=_point(kin,previous,tcp,workspace)
    via=[require_limits(kin,q).copy() for q in profile.get('via_joints_rad',[])]
    above_seed=via[-1] if via else previous
    if profile['pose_type']=='joints':
        above_q=require_limits(kin,profile['above_joints_rad']).copy()
        release_q=require_limits(kin,profile['release_joints_rad']).copy()
        above,rotation=_point(kin,above_q,tcp,workspace)
        release,release_rotation=_point(kin,release_q,tcp,workspace)
    else:
        high=np.asarray(profile['above_tcp_pose_m_rad'],float)
        low=np.asarray(profile['release_tcp_pose_m_rad'],float)
        above,release=high[:3].copy(),low[:3].copy()
        rotation,release_rotation=Rotation.from_euler('xyz',high[3:]),Rotation.from_euler('xyz',low[3:])
        above_q=kin.solve(above-rotation.apply(tcp),rotation,above_seed)
        release_q=kin.solve(release-release_rotation.apply(tcp),release_rotation,above_q)
        _point(kin,above_q,tcp,workspace);_point(kin,release_q,tcp,workspace)
    if (np.linalg.norm(above[:2]-release[:2])>.01
            or (rotation.inv()*release_rotation).magnitude()>.01
            or above[2]-release[2]<.05-1e-9):
        raise ValueError('Bin above/release must share XY/orientation and have >= 50 mm vertical clearance')
    # The transfer stays at least 50 mm above the belt, and near or above both
    # endpoint heights. These are nominal checks, not an environment collision model.
    # Do not carry an object toward the box below the above-bin height.
    # Lift to this height before starting any joint transfer.
    transit_floor=max(belt_z+.05,above[2]-.006)
    route=[]
    pickup_raise=None
    minimum=float('inf')
    if lift_tip[2]<transit_floor:
        # _make_plan validates only the pickup.  Raise to the placement carry
        # height here, from the measured post-lift joints, before crossing to
        # the taught bin route.  The first segment only needs belt clearance;
        # subsequent transfer segments must remain at the higher transit floor.
        carry_tip=lift_tip.copy();carry_tip[2]=max(transit_floor,above[2])
        carry_q=kin.solve(carry_tip-lift_rotation.apply(tcp),lift_rotation,previous)
        actual_tip,actual_rotation=_point(kin,carry_q,tcp,workspace)
        if (np.linalg.norm(actual_tip-carry_tip)>.003
                or (lift_rotation.inv()*actual_rotation).magnitude()>.03):
            raise ValueError('No verified carry-height raise after pickup lift')
        minimum=_joint_route(kin,previous,[carry_q],tcp,workspace,belt_z+.05)
        route.append(carry_q.copy());previous=carry_q
        pickup_raise=carry_tip.copy()
    transfer=via+[above_q]
    minimum=min(minimum,_joint_route(kin,previous,transfer,tcp,workspace,transit_floor))
    route.extend(transfer)
    previous=above_q
    # Preflight the move_l down and back up on the SAME IK branch.
    down=[]
    for alpha in np.linspace(0.,1.,max(10,int(np.ceil(np.linalg.norm(release-above)/.01)))+1):
        target=above*(1-alpha)+release*alpha
        q=kin.solve(target-rotation.apply(tcp),rotation,previous)
        tip,actual_rotation=_point(kin,q,tcp,workspace)
        if (np.linalg.norm(tip-target)>.003 or (actual_rotation.inv()*rotation).magnitude()>.03
                or np.max(np.abs(q-previous))>.25):
            raise ValueError('Bin descent has no continuous verified Cartesian IK path')
        down.append(q.copy());previous=q
    if np.max(np.abs(previous-release_q))>.08:
        raise ValueError('Taught release joint branch disagrees with the bin descent')
    for q in reversed(down):
        _point(kin,q,tcp,workspace)
    returning=[require_limits(kin,q).copy() for q in profile.get('return_path_joints_rad',[])]
    initial=None
    if profile.get('initial_joints_rad') is not None:
        initial=require_limits(kin,profile['initial_joints_rad']).copy()
        if not returning or np.max(np.abs(returning[-1]-initial))>1e-9:
            returning.append(initial.copy())
    return_minimum=None
    if returning:
        heights=[above[2]]+[_point(kin,q,tcp,workspace)[0][2] for q in returning]
        return_floor=max(belt_z+.05,min(heights)-.006)
        return_minimum=_joint_route(kin,above_q,returning,tcp,workspace,return_floor)
    return dict(profile_path=profile['profile_path'],profile_sha256=profile['sha256'],
        transfer_joints=[q.copy() for q in route],above=above.copy(),release=release.copy(),
        rotation=rotation,above_joints=above_q.copy(),release_joints=previous.copy(),
        return_joints=returning,initial_joints=initial,
        minimum_return_tip_z_m=return_minimum,
        lift_joints=np.asarray(lift_joints,float).copy(),minimum_transfer_tip_z_m=minimum,
        transit_floor_m=float(transit_floor),pickup_raise_tcp_m=pickup_raise,
        path_check='post-pick carry raise + nominal joint interpolation + Cartesian IK; no full collision model')


def placement_summary(plan):
    pickup_raise=plan.get('pickup_raise_tcp_m')
    return dict(profile_path=plan['profile_path'],profile_sha256=plan['profile_sha256'],
        above_tcp_m=plan['above'].tolist(),release_tcp_m=plan['release'].tolist(),
        pickup_raise_tcp_m=None if pickup_raise is None else pickup_raise.tolist(),
        flange_rpy=plan['rotation'].as_euler('xyz').tolist(),
        transfer_joints_rad=[q.tolist() for q in plan['transfer_joints']],
        release_joints_rad=plan['release_joints'].tolist(),
        return_path_joints_rad=[q.tolist() for q in plan['return_joints']],
        initial_joints_rad=(None if plan.get('initial_joints') is None
                            else plan['initial_joints'].tolist()),
        minimum_return_tip_z_m=plan['minimum_return_tip_z_m'],
        minimum_transfer_tip_z_m=plan['minimum_transfer_tip_z_m'],path_check=plan['path_check'])


def plan_fixed_bin_return(kin,actual_joints,tcp,placement,belt_z,workspace):
    """Recheck the return from measured joints after Cartesian bin retraction."""
    actual=require_limits(kin,actual_joints).copy()
    tip,rotation=_point(kin,actual,tcp,workspace)
    if (np.linalg.norm(tip-placement['above'])>.006
            or (placement['rotation'].inv()*rotation).magnitude()>.06):
        raise ValueError('Measured return start is not above the fixed bin')
    returning=[require_limits(kin,q).copy() for q in placement['return_joints']]
    updated=dict(placement)
    if returning:
        heights=[tip[2]]+[_point(kin,q,tcp,workspace)[0][2] for q in returning]
        floor=max(belt_z+.05,min(heights)-.006)
        updated['minimum_return_tip_z_m']=_joint_route(kin,actual,returning,tcp,workspace,floor)
    updated['above_joints']=actual
    return updated


def record_joint_bin_pose(path,slot,joints,tcp_pose,metadata):
    """Save an operator-selected measurement; any reteach invalidates execution."""
    if slot not in ('above','release'):
        raise ValueError('Bin teaching slot must be above or release')
    q=np.asarray(joints,float);pose=np.asarray(tcp_pose,float)
    if q.shape!=(7,) or pose.shape!=(6,) or not np.isfinite(np.r_[q,pose]).all():
        raise ValueError('Teaching requires seven radians and a finite TCP pose')
    path=Path(path)
    if path.exists():
        data=yaml.safe_load(path.read_text())
        if (not isinstance(data,dict) or data.get('schema_version')!=1
                or data.get('frame_id')!='base_link' or data.get('units')!='m_rad'):
            raise ValueError('Refusing to overwrite an incompatible bin profile')
    else:
        data=dict(schema_version=1,frame_id='base_link',units='m_rad',via_joints_rad=[],
                  above_joints_rad=[],release_joints_rad=[],above_tcp_pose_m_rad=[],release_tcp_pose_m_rad=[])
    data.update(configured=False,pose_type='joints')
    data[slot+'_joints_rad']=q.tolist()
    data[slot+'_tcp_pose_m_rad']=pose.tolist()
    data.setdefault('measurements',{})[slot]=deepcopy(metadata)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(yaml.safe_dump(data,sort_keys=False))
    temporary.replace(path)
    return data
