"""Grasp state machine for conveyor picking (pure logic, no ROS deps).

States: IDLE -> TRACK -> APPROACH -> WAIT -> DESCEND -> GRASP -> LIFT ->
PLACE -> IDLE;  any failure -> RECOVER -> IDLE;  miss >= 3 -> HALT.

Gate strategy: the arm pre-positions above a fixed grasp line on the belt,
then descends when the block's *predicted* position says it will be under the
gripper when the descend finishes (lead-compensated), verifies with the
camera each frame during DESCEND, and aborts upward on mismatch.

Motion commands are emitted once per state entry (or when the target changes
significantly); re-sending the same move each tick would restart the
interpolation on the arm.

Input track objects (msg-like): id, s, speed, speed_ok, status, height,
belt_z, centroid_x, centroid_y, u_px, v_px, stamp (float, same clock as now).
"""
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import yaw_orientations

# states
IDLE, TRACK, APPROACH, WAIT, DESCEND, GRASP, LIFT, PLACE, RECOVER, HALT = (
    'IDLE', 'TRACK', 'APPROACH', 'WAIT', 'DESCEND', 'GRASP', 'LIFT',
    'PLACE', 'RECOVER', 'HALT')


def _fold_half_pi(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


class GraspFsm:
    def __init__(self, belt, cfg):
        """belt: object with s_of(), point_of(), gate_s, d_hat.
        cfg: dict with keys (defaults):
          approach_height, descend_offset, lift_height, t_approach,
          tau_desc, k_lead, gate_tol, abort_tol, yaw_off, grasp_roll,
          grasp_pitch, fallback_width_mm, gripper_force, speed_min,
          speed_max, timeouts, miss_halt_threshold, dry_run,
          bin_pre_joints, bin_drop_pose
        """
        self.belt = belt
        c = dict(
            approach_height=0.08, descend_offset=0.04, lift_height=0.12,
            t_approach=1.5, tau_desc=0.5, k_lead=0.05,
            gate_tol=0.02, abort_tol=0.025, yaw_off=0.0,
            grasp_roll=math.pi, grasp_pitch=0.0,
            fallback_width_mm=40.0, gripper_force=1.0,
            speed_min=0.01, speed_max=0.10,
            timeout_approach=3.0, timeout_wait=5.0, timeout_descend=2.0,
            timeout_grasp=1.5, timeout_lift=3.0, timeout_place=10.0,
            miss_halt_threshold=3, dry_run=False,
            bin_pre_joints=None, bin_drop_pose=None,
            gripper_tcp=[0., 0., 0.], grasp_rpy=None,
            closing_axis_flange=[0., 0., 1.],
            observation_max_age=.2, grasp_max_age=.3, grasp_belt_clearance=.012,
        )
        c.update(cfg)
        self.cfg = c
        self.state = IDLE
        self.target_id = None
        self.grasp = None          # latest usable live grasp result
        self.grasp_used = None     # frozen at WAIT->DESCEND
        self.last_grasp_pose = None
        self.last_delta_s = None   # (s_pred - gate_s) at the grasp moment
        self.misses = 0
        self.successes = 0
        self._state_t0 = time.monotonic()
        self._last_yaw = None
        self._step = 0
        self._last_move = None     # (type, target) of last emitted move cmd
        self._log = []
        self._step_started = self._state_t0
        self._width_previous = None
        self._width_settled = None

    # ---------------- helpers ----------------

    def _enter(self, s):
        self.state = s
        self._state_t0 = time.monotonic()
        self._step = 0
        self._last_move = None
        self._step_started = self._state_t0
        self._width_previous = self._width_settled = None

    def _elapsed(self):
        return time.monotonic() - self._state_t0

    def _find_track(self, inp, tid):
        for t in inp['tracks']:
            if t.id == tid:
                return t
        return None

    def _track_valid(self, tr, inp):
        return (tr is not None and tr.status == 1 and tr.speed_ok
                and math.isfinite(tr.speed)
                and self.cfg['speed_min'] <= tr.speed <= self.cfg['speed_max']
                and math.isfinite(tr.stamp)
                and 0 <= inp['now']-tr.stamp <= self.cfg['observation_max_age'])

    def placement_valid(self):
        for name, length in [('bin_pre_joints', 7), ('bin_drop_pose', 6)]:
            value = self.cfg[name]
            if value is None or len(value) != length or not np.isfinite(value).all():
                return False
        return True

    def _fail(self, cmds, reason):
        cmds.extend([{'type': 'log', 'msg': reason}, {'type': 'estop'}])
        self.misses += 1
        self._enter(HALT)

    def _best_track(self, inp):
        """Closest-to-gate confirmed track with valid speed."""
        best, best_s = None, -1e9
        for t in inp['tracks']:
            if not self._track_valid(t, inp) or t.s > self.belt.gate_s:
                continue
            if t.speed <= 0:
                continue
            if t.speed < self.cfg['speed_min'] or t.speed > self.cfg['speed_max']:
                continue
            if t.s >= best_s:
                best, best_s = t, t.s
        return best

    def _pred_s(self, tr, inp):
        """Predicted scalar belt position right now."""
        return tr.s + tr.speed * max(inp['now'] - getattr(tr, 'motion_stamp', tr.stamp), 0.0)

    def _live_grasp(self):
        """Grasp result to plan with: frozen during DESCEND..LIFT, live else."""
        if self.grasp_used is not None:
            return self.grasp_used
        return self.grasp

    def _grasp_yaw(self):
        g = self._live_grasp()
        if g is not None and g.get('status') == 0:
            yaw = _fold_half_pi(g['yaw'] + self.cfg['yaw_off'])
        else:
            yaw = _fold_half_pi(math.atan2(self.belt.d_hat[1], self.belt.d_hat[0])
                                + self.cfg['yaw_off'])
        if self._last_yaw is not None and abs(yaw - self._last_yaw) > math.pi / 4:
            yaw = self._last_yaw  # jump guard (symmetric blocks)
        self._last_yaw = yaw
        return yaw

    def _pose(self, x, y, z, yaw):
        # Driver poses refer to the flange; input points refer to the calibrated TCP.
        if self.cfg['grasp_rpy'] is None:
            rotation = Rotation.from_euler('xyz', [self.cfg['grasp_roll'], self.cfg['grasp_pitch'], yaw])
        else:
            rotation = next(yaw_orientations(self.cfg['grasp_rpy'],
                                             self.cfg['closing_axis_flange'], yaw))
        flange = np.array([x, y, z])-rotation.apply(self.cfg['gripper_tcp'])
        return [*flange.tolist(), *rotation.as_euler('xyz').tolist()]

    def _gate_above_pose(self, tr):
        pose = self._grasp_pose(tr)
        pose[2] += self._descend_offset(tr)+self.cfg['approach_height']
        return pose

    def _descend_offset(self, tr):
        return min(self.cfg['descend_offset'], max(tr.height - 0.005, 0.005))

    def _grasp_pose(self, tr):
        """Project the object-relative grasp point onto the fixed interception line."""
        g = self._live_grasp()
        if g is not None and g.get('status') == 0:
            center = np.asarray(g['center'], float).copy()
        else:
            center = np.array([tr.centroid_x, tr.centroid_y,
                               getattr(tr, 'centroid_z', tr.belt_z+tr.height)])
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Invalid grasp point')
        center[:2] += self.belt.d_hat*(self.belt.gate_s-self.belt.s_of(center[:2]))
        center[2] = max(center[2]-self._descend_offset(tr),
                        tr.belt_z+self.cfg['grasp_belt_clearance'])
        return self._pose(*center, self._grasp_yaw())

    def _gripper_target(self):
        g = self._live_grasp()
        if g is not None and g.get('status') == 0:
            return g['width_mm'] / 1000.0
        return self.cfg['fallback_width_mm'] / 1000.0

    def _move(self, cmds, kind, target):
        """Emit a move command once per (state, target)."""
        key = (kind, tuple(round(v, 4) if isinstance(v, float) else v
                           for v in target))
        if self._last_move != key:
            self._last_move = key
            field = 'pose' if kind in ('move_p', 'move_l') else 'joints'
            cmds.append({'type': kind, field: target})

    # ---------------- tick ----------------

    def tick(self, inp):
        """inp: now, tracks (list), grasp (dict|None), arm_done (bool),
        gripper_width (m|None), joint_state (list|None).
        Returns list of command dicts."""
        cmds = []
        live = inp.get('grasp')
        self.grasp = (live if live is not None and live.get('status') == 0
                      and live.get('target_id') == self.target_id
                      and math.isfinite(live.get('stamp', float('nan')))
                      and 0 <= inp['now']-live['stamp'] <= self.cfg['grasp_max_age']
                      else None)
        tr = self._find_track(inp, self.target_id) if self.target_id is not None else None

        if self.state in (TRACK, APPROACH, WAIT, DESCEND) and not self._track_valid(tr, inp):
            self.grasp = None
            if self.state == TRACK:
                self.target_id = None
                self._enter(IDLE)
            else:
                self._fail(cmds, 'Target lost, stale or velocity invalid')
            return cmds

        if self.state == IDLE:
            best = self._best_track(inp)
            if best is not None and not self.placement_valid():
                self._fail(cmds, 'Teach valid bin_pre_joints and bin_drop_pose before conveyor picking')
                return cmds
            if best is not None:
                self._last_yaw = None
                self.target_id = best.id
                self.grasp_used = None
                self.grasp = None
                self._enter(TRACK)

        elif self.state == TRACK:
            if tr is None:
                self.target_id = None
                self._enter(IDLE)
            else:
                t_gate = (self.belt.gate_s - self._pred_s(tr, inp)) / tr.speed
                if t_gate <= self.cfg['t_approach']:
                    cmds.append({'type': 'log',
                                 'msg': f'approach trigger t_gate={t_gate:.2f}'})
                    self._enter(APPROACH)

        elif self.state == APPROACH:
            if tr is None:
                cmds.append({'type': 'log', 'msg': 'track lost in APPROACH'})
                self._enter(RECOVER)
            elif self._elapsed() > self.cfg['timeout_approach']:
                cmds.append({'type': 'log', 'msg': 'APPROACH timeout'})
                self._enter(RECOVER)
            elif self._step == 0:
                cmds.append({'type': 'gripper', 'width': .070, 'force': self.cfg['gripper_force']})
                self._move(cmds, 'move_p', self._gate_above_pose(tr))
                self._step = 1
            elif inp['arm_done'] and self._elapsed() > 0.1:
                self._enter(WAIT)

        elif self.state == WAIT:
            if tr is None:
                cmds.append({'type': 'log', 'msg': 'track lost in WAIT'})
                self._enter(RECOVER)
            elif self._elapsed() > self.cfg['timeout_wait']:
                cmds.append({'type': 'log', 'msg': 'WAIT timeout'})
                self._enter(RECOVER)
            elif self._pred_s(tr, inp) > self.belt.gate_s + self.cfg['gate_tol']:
                cmds.append({'type': 'log', 'msg': 'block passed gate'})
                self._enter(RECOVER)
            elif self.grasp is not None and self._descend_trigger(tr, inp):
                self.grasp_used = self.grasp
                self.last_grasp_pose = self._grasp_pose(tr)
                cmds.append({'type': 'log',
                             'msg': f'descend trigger s={tr.s:.3f} '
                                    f'v={tr.speed:.3f}'})
                self._enter(DESCEND)
            else:
                pass  # Hold the completed approach; do not restart it for image jitter.

        elif self.state == DESCEND:
            if tr is not None and self._elapsed() > 0.1:
                remaining = max(self.cfg['tau_desc']-self._elapsed(), 0.)
                ds = abs(self._pred_s(tr, inp)+tr.speed*remaining-self.belt.gate_s)
                if ds > self.cfg['abort_tol']:
                    cmds.append({'type': 'log',
                                 'msg': f'DESCEND abort ds={ds:.3f}'})
                    self._enter(RECOVER)
            if not cmds:
                if inp['arm_done'] and self._elapsed() > 0.2:
                    if tr is not None:
                        self.last_delta_s = (self._pred_s(tr, inp)
                                             - self.belt.gate_s)
                    self._enter(GRASP)
                elif self._elapsed() > self.cfg['timeout_descend']:
                    cmds.append({'type': 'log', 'msg': 'DESCEND timeout'})
                    self._enter(RECOVER)
                else:
                    self._move(cmds, 'move_l', self.last_grasp_pose)

        elif self.state == GRASP:
            target = self._gripper_target()
            if self.cfg['dry_run']:
                self._enter(LIFT)
            elif self._step == 0:
                cmds.append({'type': 'gripper', 'width': target,
                             'force': self.cfg['gripper_force']})
                self._step = 1
            else:
                width = inp.get('gripper_width')
                if width is not None and math.isfinite(width):
                    if self._width_previous is None or abs(width-self._width_previous) > .001:
                        self._width_settled = time.monotonic()
                    self._width_previous = width
                    settled = time.monotonic()-self._width_settled > .3
                    if self._elapsed() > .5 and settled and target+.003 < width < .067:
                        cmds.append({'type': 'log', 'msg': 'grasped'})
                        self._enter(LIFT)
                if self.state == GRASP and self._elapsed() > self.cfg['timeout_grasp']:
                    cmds.append({'type': 'log', 'msg': 'Empty or unverified grasp'})
                    self._enter(RECOVER)

        elif self.state == LIFT:
            if inp['arm_done'] and self._elapsed() > 0.3:
                self._enter(PLACE)
            elif self._elapsed() > self.cfg['timeout_lift']:
                self._enter(RECOVER)
            else:
                p = list(self.last_grasp_pose)
                p[2] += self.cfg['lift_height']
                self._move(cmds, 'move_l', p)

        elif self.state == PLACE:
            cmds += self._place_sequence(inp)

        elif self.state == RECOVER:
            cmds += self._recover_sequence(inp)

        elif self.state == HALT:
            pass

        for c in cmds:
            if c['type'] == 'log':
                self._log.append(f'[{self.state}] {c["msg"]}')
        if len(self._log) > 50:
            self._log = self._log[-50:]
        return cmds

    # ---------------- sequences ----------------

    def _descend_trigger(self, tr, inp):
        age = inp['now'] - getattr(tr, 'motion_stamp', tr.stamp)
        tau = self.cfg['tau_desc']
        lead = self.cfg['k_lead']
        # block (extrapolated past vision delay) + descend travel vs gate
        return tr.s + tr.speed * (age + tau - lead) >= self.belt.gate_s

    def _recover_sequence(self, inp):
        cmds = []
        step = self._step
        self._step += 1
        if step == 0:
            cmds.append({'type': 'log', 'msg': 'RECOVER'})
            cmds.append({'type': 'gripper', 'width': 0.070,
                         'force': self.cfg['gripper_force']})
        elif step == 1:
            # home once; give the motion time to register before waiting
            self._move(cmds, 'home', [])
            if self._elapsed() > 1.5 and inp['arm_done']:
                self._step = 2
            else:
                self._step = 1
        else:
            self.misses += 1
            self.target_id = None
            self.grasp_used = None
            if self.misses >= self.cfg['miss_halt_threshold']:
                cmds.append({'type': 'log',
                             'msg': f'HALT after {self.misses} misses'})
                self._enter(HALT)
            else:
                self._enter(IDLE)
        return cmds

    def _place_sequence(self, inp):
        cmds = []
        if not self.placement_valid():
            self._fail(cmds, 'Missing or invalid taught placement')
            return cmds
        if self._elapsed() > self.cfg['timeout_place']:
            self._fail(cmds, 'PLACE timeout')
            return cmds
        js, drop = self.cfg['bin_pre_joints'], self.cfg['bin_drop_pose']
        age = time.monotonic()-self._step_started
        if self._step == 0:
            self._move(cmds, 'move_j', js)
            self._step, self._step_started = 1, time.monotonic()
        elif self._step == 1 and inp['arm_done'] and age > .1:
            self._move(cmds, 'move_l', drop)
            self._step, self._step_started = 2, time.monotonic()
        elif self._step == 2 and inp['arm_done'] and age > .1:
            cmds.append({'type': 'gripper', 'width': .070, 'force': self.cfg['gripper_force']})
            self._step, self._step_started = 3, time.monotonic()
        elif self._step == 3:
            width = inp.get('gripper_width')
            if age > .3 and width is not None and math.isfinite(width) and width >= .067:
                self.successes += 1
                self.misses = 0
                self.target_id = self.grasp_used = self.grasp = None
                self._enter(IDLE)
        return cmds

    def reset_halt(self):
        """Manual reset after HALT."""
        self.misses = 0
        self.target_id = None
        self.grasp = self.grasp_used = None
        self._enter(IDLE)
