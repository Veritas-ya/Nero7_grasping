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


def draw_speed_overlay(image, tracks, pixel_scale=1., error=None):
    """Keep each label next to its own centroid and inside the image bounds."""
    result = image.copy()
    height, width = result.shape[:2]
    if error:
        # A rejected frame must never display a previous valid velocity.
        for line, text in enumerate(('Detection unavailable - speed unavailable', str(error))):
            font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (text_width, _), _ = cv2.getTextSize(text, font, scale, thickness)
            scale *= min(1., max(1, width-24)/max(1, text_width))
            cv2.putText(result, text, (12, 28+line*28), font, scale,
                        (0, 0, 255), thickness, cv2.LINE_AA)
        return result
    placed = []
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
        left = max(4, min(u-text_width-12, width-text_width-4))
        below = max(text_height+4, min(v+text_height+12, height-baseline-4))
        candidates = [(x, y), (x, below), (left, y), (left, below)]
        step = text_height+baseline+10
        for offset in range(1, max(1, height//step)):
            for direction in (-1, 1):
                shifted_y = max(text_height+4, min(y+direction*offset*step, height-baseline-4))
                candidates.append((x, shifted_y))
        for candidate_x, candidate_y in candidates:
            rectangle = (candidate_x-3, candidate_y-text_height-3,
                         candidate_x+text_width+3, candidate_y+baseline+3)
            if all(rectangle[2]+4 < other[0] or other[2]+4 < rectangle[0]
                   or rectangle[3]+4 < other[1] or other[3]+4 < rectangle[1] for other in placed):
                x, y = candidate_x, candidate_y
                break
        placed.append((x-3, y-text_height-3, x+text_width+3, y+baseline+3))
        cv2.line(result, (u, v), (int(np.clip(u, x, x+text_width)),
                                 int(np.clip(v, y-text_height, y+baseline))), color, 1)
        cv2.rectangle(result, (x - 3, y - text_height - 3),
                      (x + text_width + 3, y + baseline + 3), (25, 25, 25), -1)
        cv2.putText(result, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    if not tracks:
        cv2.putText(result, 'No tracked blocks - speed unavailable', (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
    return result
