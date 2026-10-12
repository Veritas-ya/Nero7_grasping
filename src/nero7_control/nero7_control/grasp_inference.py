"""GR-ConvNet inference wrapper: async GPU thread, latest-only requests.

Input: full-frame RGB + aligned depth (color grid) + block centroid pixel.
Output: GraspCandidate in base_link frame (center, yaw, width_mm, score).

The camera math is contained here: pixel backprojection uses the color
intrinsics; the grasp angle is converted to a base-frame yaw by
backprojecting the angle direction vector, which absorbs any top-down
flip/mirror of the camera mount automatically.
"""
import glob
import os
import threading
import time

import numpy as np
import torch

from .grconvnet.post_process import post_process_output
from .grconvnet.grasp import detect_grasps
from .inference_geometry import optical_rays

CROP = 224
# trained with RGB: img/255 - mean; depth: img/255 - mean (see CameraData)
# real depth arrives in mm as uint16 -> scale so values land in the
# 0..255-ish range the network saw during training (mm/4 for 0..1m)
DEPTH_MM_SCALE = 4.0


class GraspInference:
    def __init__(self, model_path, device='cuda', max_rate=10.0,
                 q_threshold=0.25, target_radius_px=60.0):
        self._device = device
        self._max_dt = 1.0 / max_rate
        self._q_threshold = q_threshold
        self._target_radius_px = float(target_radius_px)
        if not 0 < self._target_radius_px <= CROP/2:
            raise ValueError('target_radius_px must be within (0, crop_size/2]')
        self._last_req_t = 0.0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._request = None
        self._request_seq = 0
        self._result = None
        self._running = True

        model_path = _resolve_model_path(model_path)
        _register_legacy_modules()
        self._model = torch.load(model_path, weights_only=False, map_location=device)
        self._model.eval()
        self._model = self._model.to(device)
        # warmup
        with torch.no_grad():
            self._model.predict(torch.zeros((1, 4, CROP, CROP), device=device))
        print(f'[grasp_inference] model loaded: {model_path}')

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------------- public API ----------------

    def request(self, rgb, depth_mm, u, v, K, stamp_sec, t_base_cam, belt_z=None, target_id=None, distortion=(), distortion_model='plumb_bob'):
        """Submit an inference request for the block at pixel (u, v).

        Full images in, crop is made by the worker.  Older pending requests
        are dropped (latest-only).
        """
        with self._cond:
            self._request_seq += 1
            self._request = dict(rgb=rgb, depth=depth_mm, u=u, v=v, K=K,
                                 stamp=stamp_sec, tf=t_base_cam,
                                 belt_z=belt_z,
                                 target_id=target_id,
                                 distortion=tuple(distortion), distortion_model=distortion_model,
                                 seq=self._request_seq)
            self._cond.notify()

    def latest(self, max_age=0.2):
        """Latest finished result as a dict, or None if stale/none."""
        with self._lock:
            if self._result is None:
                return None
            if time.monotonic() - self._result['finish_t'] > max_age:
                return None
            return self._result

    def shutdown(self):
        self._running = False
        with self._cond:
            self._cond.notify()

    # ---------------- worker ----------------

    def _loop(self):
        while self._running:
            with self._cond:
                while self._running and self._request is None:
                    self._cond.wait()
                if not self._running:
                    return
                req, self._request = self._request, None
                seq = req['seq']
            # skip if a newer request superseded this one (latest-only)
            with self._lock:
                if seq != self._request_seq:
                    continue
                now = time.monotonic()
                if now - self._last_req_t < self._max_dt:
                    continue
                self._last_req_t = now
            try:
                cand = self._infer(req)
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                print(f'[grasp_inference] infer failed: {e}')
                cand = dict(status=2, error=str(e), stamp=req['stamp'],
                            finish_t=time.monotonic())
            cand['finish_t'] = time.monotonic()
            cand['target_id'] = req.get('target_id')
            with self._lock:
                self._result = cand

    def _infer(self, req):
        rgb, depth = req['rgb'], req['depth']
        crop_rgb, crop_d, crop_u0, crop_v0 = make_crop(
            rgb, depth, req['u'], req['v'], size=CROP)

        # normalize depth: mm -> scaled float -> /255 - mean (as trained)
        d = crop_d.astype(np.float32) / DEPTH_MM_SCALE / 255.0
        d -= d.mean()
        r = crop_rgb.astype(np.float32) / 255.0
        r -= r.mean(axis=(0, 1), keepdims=True)
        x = np.concatenate([d[None, :, :], r.transpose(2, 0, 1)], axis=0)
        x = torch.from_numpy(x[None, ...].astype(np.float32)).to(self._device)

        with torch.no_grad():
            pred = self._model.predict(x)
        q_img, ang_img, width_img = post_process_output(
            pred['pos'], pred['cos'], pred['sin'], pred['width'])

        if req.get('belt_z') is not None:
            # Select network peaks on the detected object, not on belt/background.
            vv, uu = np.mgrid[:CROP, :CROP]
            us, vs = uu+crop_u0, vv+crop_v0
            dm = crop_d.astype(float)/1000
            K = req['K']
            rays = optical_rays(np.column_stack([us.ravel(), vs.ravel()]), K,
                                req.get('distortion', ()), req.get('distortion_model', 'plumb_bob'))
            cam = rays.reshape(CROP, CROP, 3)*dm[..., None]
            tf = req['tf']
            bz = (cam @ tf['R'].T + tf['t'])[..., 2]
            mask = (np.isfinite(dm) & (dm > 0)
                    & ((us-req['u'])**2+(vs-req['v'])**2 < self._target_radius_px**2)
                    & (bz > req['belt_z']+.005) & (bz < req['belt_z']+.12))
            q_img = np.where(mask, q_img, 0)
        grasps = detect_grasps(q_img, ang_img, width_img)
        if not grasps:
            return dict(status=1, stamp=req['stamp'])
        g = grasps[0]  # highest q peak

        # grasp point in the full color grid
        cy, cx = g.center  # [y, x] within the crop
        score = float(q_img[cy, cx])
        if score < self._q_threshold:
            return dict(status=1, stamp=req['stamp'], score=score)
        u = crop_u0 + cx
        v = crop_v0 + cy
        K = req['K']
        # depth at the grasp point (mm -> m), from the full aligned depth
        z = float(depth[int(round(v)), int(round(u))]) / 1000.0
        if not np.isfinite(z) or z <= 0:
            # Never substitute conveyor/background depth for an invalid grasp.
            vi, ui = int(round(v)), int(round(u))
            dv = depth[max(0, vi-2):vi+3, max(0, ui-2):ui+3].astype(np.float32)
            dv[dv <= 0] = np.nan
            if not np.isfinite(dv).any():
                return dict(status=1, stamp=req['stamp'])
            z = float(np.nanmedian(dv))
            z = z / 1000.0 if np.isfinite(z) else 0.0
            if z <= 0:
                return dict(status=1, stamp=req['stamp'])

        # backproject center + direction vector to 3D in camera frame
        distortion = req.get('distortion', ())
        model = req.get('distortion_model', 'plumb_bob')
        p0 = _backproject(u, v, z, K, distortion, model)
        ang = g.angle
        # Grasp.angle is anticlockwise: image rows increase downwards.
        u1, v1 = u + 12.0 * np.cos(ang), v - 12.0 * np.sin(ang)
        p1 = _backproject(u1, v1, z, K, distortion, model)
        t = req['tf']
        R, tvec = t['R'], t['t']
        p0b = R @ p0 + tvec
        p1b = R @ p1 + tvec
        d = p1b - p0b
        yaw = float(np.arctan2(d[1], d[0]))
        # fold two-finger yaw into [-pi/2, pi/2)
        yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2

        # width: crop pixels -> mm at block depth, with grasp factor
        width_mm = float(g.length)*np.linalg.norm(p1-p0)/12.*1000.
        raw_width_mm = width_mm
        width_mm = float(np.clip(width_mm * 0.85, 15.0, 60.0))

        return dict(status=0, stamp=req['stamp'],
                    center=p0b, yaw=yaw, width_mm=width_mm,
                    raw_width_mm=raw_width_mm,
                    score=score, u_px=float(u), v_px=float(v))


def _resolve_model_path(path):
    """Accept a checkpoint file, or a directory of epoch_*_iou_* files."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, 'epoch_*'))
        if not files:
            raise FileNotFoundError(f'no epoch files in {path}')

        def _iou(fn):
            name = os.path.basename(fn)
            return float(name.split('iou_')[-1]) if 'iou_' in name else 0.0
        path = max(files, key=_iou)
    return path


def _register_legacy_modules():
    """The pretrained checkpoints were pickled in the upstream repo layout
    (``inference.models.*``).  Alias our copied modules so torch.load can
    resolve the classes."""
    import sys
    import types
    from .grconvnet import grconvnet3, grasp_model
    for name in ('inference', 'inference.models'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules.setdefault('inference.models.grconvnet3', grconvnet3)
    sys.modules.setdefault('inference.models.grasp_model', grasp_model)


def make_crop(rgb, depth, u, v, size=CROP):
    """Center crop at (u, v) with clamp + edge padding.

    Returns (rgb_crop, depth_crop, crop_u0, crop_v0) where (crop_u0, crop_v0)
    is the full-image coordinate of crop pixel (0, 0).
    """
    h, w = rgb.shape[:2]
    half = size // 2
    u0, v0 = int(round(u)) - half, int(round(v)) - half
    u1, v1 = u0 + size, v0 + size
    pad_u0, pad_v0 = max(0, -u0), max(0, -v0)
    pad_u1, pad_v1 = max(0, u1 - w), max(0, v1 - h)
    su0, sv0 = max(0, u0), max(0, v0)
    su1, sv1 = min(w, u1), min(h, v1)
    rc = rgb[sv0:sv1, su0:su1]
    dc = depth[sv0:sv1, su0:su1]
    if pad_u0 or pad_v0 or pad_u1 or pad_v1:
        rc = np.pad(rc, ((pad_v0, pad_v1), (pad_u0, pad_u1), (0, 0)),
                    mode='edge')
        dc = np.pad(dc, ((pad_v0, pad_v1), (pad_u0, pad_u1)), mode='edge')
    return rc, dc, u0, v0


def _backproject(u, v, z, K, distortion=(), distortion_model='plumb_bob'):
    return optical_rays([[u, v]], K, distortion, distortion_model)[0]*z
