"""Per-object visual speed labels, using the measured XY motion estimate."""
import cv2
import numpy as np

from .block_tracker import STATUS_CONFIRMED, STATUS_LOST


def speed_label(track):
    """Do not present an unqualified or lost estimate as a live measurement."""
    prefix = f'ID {track.id}  speed='
    if track.status == STATUS_LOST:
        return prefix + '-- (LOST)'
    if not track.speed_valid:
        return prefix + '-- (WAITING)'
    velocity = np.asarray(track.velocity, dtype=float)[:2]
    if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
        return prefix + '-- (INVALID)'
    return prefix + f'{np.linalg.norm(velocity):.3f} m/s'


def draw_speed_overlay(image, tracks, pixel_scale=1.):
    """Keep each label next to its own centroid and inside the image bounds."""
    result = image.copy()
    height, width = result.shape[:2]
    for track in tracks:
        u, v = int(round(track.u_px*pixel_scale)), int(round(track.v_px*pixel_scale))
        color = ((0, 255, 0) if track.status == STATUS_CONFIRMED else
                 (0, 0, 255) if track.status == STATUS_LOST else (0, 200, 255))
        cv2.circle(result, (u, v), 8, color, 2)
        text = speed_label(track)
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        # Narrow image streams still need a complete speed and unit.
        if text_width + 8 > width:
            scale *= max(1, width - 8) / text_width
            (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(4, min(u + 12, width - text_width - 4))
        y = max(text_height + 4, min(v - 12, height - baseline - 4))
        cv2.rectangle(result, (x - 3, y - text_height - 3),
                      (x + text_width + 3, y + baseline + 3), (25, 25, 25), -1)
        cv2.putText(result, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    if not tracks:
        cv2.putText(result, 'No tracked blocks - speed unavailable', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return result
