"""Shared calibration math and board detection for touch_calib / drag_calib."""
import os

import cv2
import numpy as np
import yaml

DEFAULT_CALIB = ('/home/ubuntu22/Nero7_grasping/src/nero7_bringup/config/'
                 'calibration.yaml')


def quat_R(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x *= -1 if R[2, 1] - R[1, 2] < 0 else 1
    y *= -1 if R[0, 2] - R[2, 0] < 0 else 1
    z *= -1 if R[1, 0] - R[0, 1] < 0 else 1
    return np.array([w, x, y, z])


def axis_angle_R(v, ang):
    v = np.asarray(v, float)
    v = v / np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def solve_tcp(flange_poses):
    """4-point TCP: tool tip position in the flange frame (least squares).

    flange_poses: list of 4x4 (flange in base), tip touching one fixed point.
    """
    A, b = [], []
    for T in flange_poses:
        R, t = T[:3, :3], T[:3, 3]
        A.append(np.hstack([R, -np.eye(3)]))
        b.append(-t)
    A, b = np.vstack(A), np.concatenate(b)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    tcp, p_base = x[:3], x[3:]
    resid = float(np.linalg.norm(A @ x - b))
    return tcp, p_base, resid


def solve_board(pts_base, pts_cam):
    """Kabsch: T_base_cam (p_base = R p_cam + t) from point correspondences."""
    pts_base = np.asarray(pts_base)
    pts_cam = np.asarray(pts_cam)
    mb, mc = pts_base.mean(axis=0), pts_cam.mean(axis=0)
    H = (pts_cam - mc).T @ (pts_base - mb)
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mb - R @ mc
    resid = np.linalg.norm((R @ pts_cam.T).T + t - pts_base, axis=1)
    return R, t, resid


def detect_board(color, depth, K, board=(11, 8), square=0.02, D=None,
                 d2c_offset=(0, 0)):
    """Detect the chessboard; return (Nx3 corners in camera optical frame,
    Nx2 pixel corners, N original grid indices) or (None, None, None).

    Corner 3D positions come from intersecting the corner pixel rays with a
    RANSAC plane fit to the depth pixels INSIDE the board region (robust to
    per-corner depth noise and board-edge artifacts).

    ``d2c_offset``: (du, dv) correction for the driver's depth-to-color
    registration (constant pixel offset of the registered depth).  The depth
    is shifted by this amount before use."""
    if d2c_offset != (0, 0):
        depth = _shift_depth(depth, d2c_offset[0], d2c_offset[1])
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, board, None)
    if not ok:
        return None, None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    pts2d = corners.reshape(-1, 2)

    # robust board plane from interior depth pixels
    quad = pts2d[[0, board[0] - 1,
                  (board[1] - 1) * board[0],
                  board[0] * board[1] - 1]]
    mask = np.zeros(depth.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((15, 15), np.uint8))  # keep well inside
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs].astype(np.float32) / 1000.0
    valid = np.isfinite(z) & (z > 0.1) & (z < 2.0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if z.size < 200:
        return None, None, None
    pts3d = np.column_stack([(xs - K[0, 2]) * z / K[0, 0],
                             (ys - K[1, 2]) * z / K[1, 1], z])
    # RANSAC plane fit (n, d): n.p + d = 0
    rng = np.random.default_rng(0)
    best_n, best_d, best_in = None, None, -1
    for _ in range(200):
        idx = rng.choice(len(pts3d), 3, replace=False)
        p0, p1, p2 = pts3d[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p0
        inl = np.abs(pts3d @ n + d) < 0.004
        if inl.sum() > best_in:
            best_in = inl.sum()
            best_n, best_d = n, d
    if best_n is None or best_in < 0.5 * len(pts3d):
        return None, None, None

    # undistort corner pixels -> rays -> intersect the plane
    if D is not None and np.any(D):
        pts2d_u = cv2.undistortPoints(pts2d.reshape(-1, 1, 2), K, D)
        pts2d_u = pts2d_u.reshape(-1, 2)
        pts2d_u = pts2d_u * np.array([K[0, 0], K[1, 1]]) + \
            np.array([K[0, 2], K[1, 2]])
    else:
        pts2d_u = pts2d
    rays = np.column_stack([(pts2d_u[:, 0] - K[0, 2]) / K[0, 0],
                            (pts2d_u[:, 1] - K[1, 2]) / K[1, 1],
                            np.ones(len(pts2d_u))])
    lam = -best_d / (rays @ best_n)
    pts3d_c = rays * lam[:, None]
    return pts3d_c, pts2d, np.arange(board[0] * board[1])


def pick_touch_order(orig_idx, board=(11, 8)):
    """Choose touch corners: the 4 grid extremes + center, mapped to the
    nearest valid corners.  Returns indices into the valid-corner arrays
    (same numbering as the annotated image)."""
    bw, bh = board
    targets = np.array([[0, 0], [bw - 1, 0], [0, bh - 1], [bw - 1, bh - 1],
                        [(bw - 1) // 2, (bh - 1) // 2]], float)  # col, row
    grid = np.column_stack([orig_idx % bw, orig_idx // bw]).astype(float)
    order = []
    for t in targets:
        d = np.linalg.norm(grid - t, axis=1)
        order.append(int(np.argmin(d)))
    # de-duplicate while keeping order
    seen, out = set(), []
    for i in order:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _shift_depth(dep, du, dv):
    """Shift the depth image by (du, dv) pixels; edges become NaN.

    Corrected depth at (u,v) = original depth at (u+du, v+dv)."""
    out = np.full_like(dep, np.nan, dtype=np.float32)
    H, W = dep.shape
    sv0, sv1 = max(0, dv), min(H, H + dv)
    su0, su1 = max(0, du), min(W, W + du)
    dv0, dv1 = max(0, -dv), min(H, H - dv)
    du0, du1 = max(0, -du), min(W, W - du)
    out[dv0:dv1, du0:du1] = dep[sv0:sv1, su0:su1]
    return out


def save_annotated(color, pts2d, touch_order=None, path='/tmp/touch_corners.png'):
    img = color.copy()
    for i, (u, v) in enumerate(pts2d):
        cv2.circle(img, (int(round(u)), int(round(v))), 5, (0, 255, 0), 2)
        cv2.putText(img, str(i), (int(round(u)) + 8, int(round(v)) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if touch_order:
        for k, i in enumerate(touch_order):
            u, v = pts2d[i]
            cv2.circle(img, (int(round(u)), int(round(v))), 18, (0, 255, 255), 3)
            cv2.putText(img, f'T{k}', (int(round(u)) - 20, int(round(v)) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(path, img)
    return path


def write_calib_file(path, update):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    data.update(update)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(data, f)
