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

    # ---------------- helpers ----------------

    def _enter(self, s):
        self.state = s
        self._state_t0 = time.monotonic()
        self._step = 0
        self._last_move = None

    def _elapsed(self):
        return time.monotonic() - self._state_t0

    def _find_track(self, inp, tid):
        for t in inp['tracks']:
            if t.id == tid:
                return t
        return None

    def _best_track(self, inp):
        """Closest-to-gate confirmed track with valid speed."""
        best, best_s = None, -1e9
        for t in inp['tracks']:
            if not (t.status == 1 and t.speed_ok and t.speed > 0):
                continue
            if t.speed < self.cfg['speed_min'] or t.speed > self.cfg['speed_max']:
                continue
            if t.s >= best_s:
                best, best_s = t, t.s
        return best

    def _pred_s(self, tr, inp):
        """Predicted scalar belt position right now."""
        return tr.s + tr.speed * max(inp['now'] - tr.stamp, 0.0)

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
        return [x, y, z, self.cfg['grasp_roll'], self.cfg['grasp_pitch'], yaw]

    def _gate_above_pose(self, tr):
        gx, gy = self.belt.point_of(self.belt.gate_s)
        z = tr.belt_z + tr.height + self.cfg['approach_height']
        return self._pose(gx, gy, z, self._grasp_yaw())

    def _descend_offset(self, tr):
        return min(self.cfg['descend_offset'], max(tr.height - 0.005, 0.005))

    def _grasp_pose(self, tr):
        """Grasp target pose at the gate."""
        g = self._live_grasp()
        if g is not None and g.get('status') == 0:
            c = g['center']
            z = c[2] - self._descend_offset(tr)
            return self._pose(float(c[0]), float(c[1]), float(z), self._grasp_yaw())
        # fallback: block centroid, top-down
        z = tr.belt_z + tr.height - self._descend_offset(tr)
        return self._pose(tr.centroid_x, tr.centroid_y, z, self._grasp_yaw())

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
        if live is not None:
            self.grasp = live
        tr = self._find_track(inp, self.target_id) if self.target_id is not None else None

        if self.state == IDLE:
            best = self._best_track(inp)
            if best is not None:
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
            elif inp['arm_done'] and self._elapsed() > 0.1:
                self._enter(WAIT)
            else:
                self._move(cmds, 'move_p', self._gate_above_pose(tr))

        elif self.state == WAIT:
            if tr is None:
                cmds.append({'type': 'log', 'msg': 'track lost in WAIT'})
                self._enter(RECOVER)
            elif self._elapsed() > self.cfg['timeout_wait']:
                cmds.append({'type': 'log', 'msg': 'WAIT timeout'})
                self._enter(RECOVER)
            elif self._pred_s(tr, inp) > self.belt.gate_s + self.cfg['gate_tol']:
                cmds.append({'type': 'log', 'msg': 'block passed gate'})
                self.misses += 1
                self._enter(RECOVER)
            elif self._descend_trigger(tr, inp):
                self.grasp_used = self.grasp
                self.last_grasp_pose = self._grasp_pose(tr)
                cmds.append({'type': 'log',
                             'msg': f'descend trigger s={tr.s:.3f} '
                                    f'v={tr.speed:.3f}'})
                self._enter(DESCEND)
            else:
                # keep aiming at the gate; re-aim when the yaw refines
                self._move(cmds, 'move_p', self._gate_above_pose(tr))

        elif self.state == DESCEND:
            if tr is not None and self._elapsed() > 0.1:
                ds = abs(self._pred_s(tr, inp) - self.belt.gate_s)
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
                self._enter(LIFT)  # never close in dry run
            elif self._elapsed() < 0.1:
                cmds.append({'type': 'gripper', 'width': target,
                             'force': self.cfg['gripper_force']})
            else:
                w = inp.get('gripper_width')
                if w is not None and w < 0.070:  # feedback valid and closed-ish
                    if w > target * 1.1:         # stopped on the block
                        cmds.append({'type': 'log', 'msg': 'grasped'})
                        self._enter(LIFT)
                    else:                        # closed on air
                        cmds.append({'type': 'log', 'msg': 'empty grasp'})
                        self._enter(RECOVER)
                elif self._elapsed() > self.cfg['timeout_grasp']:
                    cmds.append({'type': 'log', 'msg': 'GRASP timeout'})
                    self._enter(RECOVER)
                else:
                    cmds.append({'type': 'gripper', 'width': target,
                                 'force': self.cfg['gripper_force']})

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
        age = inp['now'] - tr.stamp
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
        step = self._step
        self._step += 1
        js, drop = self.cfg['bin_pre_joints'], self.cfg['bin_drop_pose']
        if step == 0:
            if js is None:
                self._step = 2
            else:
                cmds.append({'type': 'move_j', 'joints': js})
        elif step == 1:
            if inp['arm_done']:
                self._step = 2
            else:
                cmds.append({'type': 'move_j', 'joints': js})
        elif step == 2:
            if drop is None:
                self._step = 4
            else:
                cmds.append({'type': 'move_l', 'pose': drop})
        elif step == 3:
            if inp['arm_done']:
                self._step = 4
            else:
                cmds.append({'type': 'move_l', 'pose': drop})
        elif step == 4:
            cmds.append({'type': 'gripper', 'width': 0.070,
                         'force': self.cfg['gripper_force']})
            cmds.append({'type': 'log', 'msg': 'released'})
            self.successes += 1
            self.target_id = None
            self.grasp_used = None
            self.grasp = None
            self._enter(IDLE)
        return cmds

    def reset_halt(self):
        """Manual reset after HALT."""
        self.misses = 0
        self.target_id = None
        self._enter(IDLE)
