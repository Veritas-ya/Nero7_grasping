"""Latency, low-FPS velocity and bounded ID recovery scenarios."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from nero7_perception.block_tracker import BlockTracker, STATUS_LOST
from nero7_perception.conveyor_model import BeltModel, BeltPlane
from nero7_perception.frame_selection import newest_pair
from nero7_perception.geometry import CameraGeometry
from nero7_perception.base_plane import BaseBeltPlane


def det(t, shift=0.):
    return dict(u=400., v=400., centroid_base=np.array([-.36+shift, .3-.028*t, -.06]),
                belt_z=-.1, height=.04, width_px=30., area_px=800.)


def fitted():
    tracker = BlockTracker(BeltModel([0., -1.]), {})
    for i in range(61):
        tracker.update([det(i/30)], i/30)
    assert tracker.tracks[0].speed_valid
    return tracker


@pytest.mark.parametrize('fps', [8., 10., 15., 30.])
def test_real_low_fps_observations_qualify_without_faking_samples(fps):
    tracker = BlockTracker(BeltModel([0., -1.]), {})
    ids = set()
    for i in range(int(fps*3)+1):
        tracks = tracker.update([det(i/fps)], i/fps)
        ids.add(tracks[0].id)
    assert ids == {0}
    assert tracks[0].speed_valid
    assert tracks[0].speed == pytest.approx(.028, abs=.001)


def test_confirmed_id_survives_more_than_six_missed_frames_but_stale_speed_is_invalid():
    tracker = fitted()
    track = tracker.tracks[0]
    count = track.motion.accepted_count
    for i in range(1, 10):
        tracker.update([], 2.+i/30)
    assert tracker.tracks[0].id == track.id and track.status == STATUS_LOST
    assert not track.speed_valid and track.motion.accepted_count == count
    tracks = tracker.update([det(2.+10/30)], 2.+10/30)
    assert len(tracks) == 1 and tracks[0].id == track.id
    assert not tracks[0].speed_valid  # A long observation gap must requalify.


def test_expired_object_is_not_revived_as_a_valid_old_velocity():
    tracker = fitted()
    old_id = tracker.tracks[0].id
    assert tracker.update([], 2.51) == []
    tracks = tracker.update([det(2.55)], 2.55)
    assert tracks[0].id != old_id and not tracks[0].speed_valid


def test_preview_identity_retention_requalifies_speed_after_half_second_gap():
    tracker = BlockTracker(BeltModel([0., -1.]), {}, lost_timeout=1.)
    for i in range(61):
        tracks = tracker.update([det(0.)], i/30)
    tid = tracks[0].id
    assert tracks[0].speed_valid
    tracks = tracker.update([det(0.)], 2.65)
    assert len(tracks) == 1 and tracks[0].id == tid
    assert not tracks[0].speed_valid
    for i in range(1, 20):
        tracks = tracker.update([det(0.)], 2.65+i/30)
    assert tracks[0].id == tid and tracks[0].speed_valid
    assert tracker.update([], 4.5) == []


def test_coherent_rejected_centroid_shift_keeps_id_but_resets_velocity():
    tracker = fitted()
    old_id = tracker.tracks[0].id
    for i in range(1, 4):
        tracks = tracker.update([det(2.+i/30, shift=.012)], 2.+i/30)
    assert len(tracks) == 1 and tracks[0].id == old_id
    assert tracks[0].reacquisitions == 1 and not tracks[0].speed_valid
    assert len(tracks[0].motion.accepted_stamps) == 3
    for i in range(4, 31):
        tracks = tracker.update([det(2.+i/30, shift=.012)], 2.+i/30)
    assert tracks[0].id == old_id and tracks[0].speed_valid


def test_single_outlier_never_changes_observation_or_speed_history():
    tracker = fitted()
    track = tracker.tracks[0]
    previous = track.observed_centroid.copy()
    count = track.motion.accepted_count
    tracker.update([det(2.+1/30, shift=.012)], 2.+1/30)
    np.testing.assert_array_equal(track.observed_centroid, previous)
    assert track.motion.accepted_count == count and track.reacquisitions == 0


def test_second_object_near_matched_track_receives_its_own_id():
    tracker = fitted()
    tracks = tracker.update([det(2.+1/30), det(2.+1/30, shift=.020)], 2.+1/30)
    assert len(tracks) == 2 and len({t.id for t in tracks}) == 2


@pytest.mark.parametrize('reverse', [False, True])
def test_global_assignment_preserves_both_ids_when_greedy_match_strands_one(reverse):
    tracker = BlockTracker(BeltModel([0., -1.]), {})
    first, second = det(0.), det(0., shift=.008)
    tracker.update([first, second], 10.)
    observations = [det(0., shift=.001), det(0., shift=-.007)]
    if reverse:
        observations.reverse()
    tracks = tracker.update(observations, 10.001)
    assert len(tracks) == 2 and tracker._next_id == 2
    assert [t.id for t in tracks] == [0, 1]
    assert all(t.misses == 0 for t in tracks)
    assert tracks[0].observed_centroid[0] == pytest.approx(first['centroid_base'][0]-.007)
    assert tracks[1].observed_centroid[0] == pytest.approx(first['centroid_base'][0]+.001)


def stamp(t):
    ns = int(round(t*1e9))
    seconds, nanoseconds = divmod(ns, 1_000_000_000)
    return NS(sec=seconds, nanosec=nanoseconds)


class DepthBuffer:
    def __init__(self, times):
        self.frames = [NS(stamp=stamp(t)) for t in times]

    def closest(self, target, delta, after):
        time = target.sec+target.nanosec*1e-9
        usable = [f for f in self.frames if f.stamp.sec+f.stamp.nanosec*1e-9 > after]
        if not usable:
            return None
        frame = min(usable, key=lambda f: abs(f.stamp.sec+f.stamp.nanosec*1e-9-time))
        return frame if abs(frame.stamp.sec+frame.stamp.nanosec*1e-9-time) <= delta else None


def test_backlog_selects_newest_unused_pair_and_does_not_reprocess_depth():
    colors = [(i, stamp(t), 'optical') for i, t in enumerate([10., 10.03, 10.06])]
    depth = DepthBuffer([10., 10.03, 10.06])
    pair = newest_pair(colors, depth, .02, -np.inf, -np.inf, 10.08, .2)
    assert pair[0] == 2
    assert newest_pair(colors, depth, .02, 10.06, 10.06, 10.08, .2) is None


def test_stale_or_unsynchronized_pairs_are_not_relabelled_fresh():
    colors = [(0, stamp(10.), 'optical')]
    assert newest_pair(colors, DepthBuffer([10.]), .02, -np.inf, -np.inf, 10.3, .2) is None
    assert newest_pair(colors, DepthBuffer([10.08]), .02, -np.inf, -np.inf, 10.1, .2) is None


def test_bounded_legacy_ransac_refines_a_tilted_plane_with_objects():
    v, u = np.indices((120, 160))
    truth = .9+.0002*u-.0001*v
    depth = truth.copy()
    depth[30:65, 50:100] -= .04
    depth[0:10] = np.nan
    plane = BeltPlane(max_points=1500)
    assert plane.fit(depth)
    assert np.max(np.abs(plane.plane_z(u, v)-truth)) < .001


def test_plane_height_cache_updates_after_transform_change():
    camera = CameraGeometry(np.array([[500., 0, 20.], [0, 500., 15.], [0, 0, 1.]]),
                            40, 30, 'optical')
    plane = BaseBeltPlane([0., 0., 1.], -.5, 'fixed', {})
    depth = np.ones((30, 40))
    transform = np.eye(4)
    first = plane.height_image(depth, camera, transform)
    transform[2, 3] = .1
    changed = plane.height_image(depth, camera, transform)
    np.testing.assert_allclose(changed-first, .1, atol=1e-12)


@pytest.mark.parametrize('filename', ['perception.yaml', 'camera_speed_preview.yaml'])
def test_source_configuration_resolves_shared_assets_without_a_machine_home(filename):
    from pathlib import Path
    import yaml
    from nero7_perception.detection import load_detection_params
    config = Path(__file__).resolve().parents[1]/'config'/filename
    configured = yaml.safe_load(config.read_text())['block_detector']['ros__parameters']
    params = load_detection_params(config)
    shared = config.parents[2]/'nero7_bringup/config'
    for name, resource in [('belt_file', 'belt.yaml'), ('calibration_file', 'calibration.yaml')]:
        assert not configured.get(name)  # Resolve resources relative to this checkout.
        assert Path(params[name]).resolve() == (shared/resource).resolve()
        assert Path(params[name]).is_file()
