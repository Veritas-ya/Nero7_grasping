"""Select a fresh, unused RGB-D pair without draining a backlog."""
import math


def stamp_seconds(stamp):
    return stamp.sec+stamp.nanosec*1e-9


def newest_pair(colors, depth_source, max_delta, last_color, last_depth, now, max_age):
    # Image queues may contain older pairs after a slow computation. Processing
    # them all increases latency; discard them and prefer the newest usable pair.
    for color, stamp, frame_id in reversed(list(colors)):
        color_time = stamp_seconds(stamp)
        if (not math.isfinite(color_time) or color_time <= last_color
                or not -max_delta <= now-color_time <= max_age):
            continue
        frame = depth_source.closest(stamp, max_delta, last_depth)
        if frame is None:
            continue
        depth_time = stamp_seconds(frame.stamp)
        if depth_time > last_depth and -max_delta <= now-depth_time <= max_age:
            return color, stamp, frame_id, frame
    return None
