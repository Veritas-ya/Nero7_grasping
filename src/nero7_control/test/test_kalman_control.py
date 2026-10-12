"""Physical timing and uncertainty checks for conveyor interception helpers."""
from types import SimpleNamespace

import numpy as np
import pytest

from nero7_control.conveyor_prediction import (
    capture_bounds,
    forecast_motion,
    motion_at_contact,
    stage_window,
)


def track(**changes):
    values = dict(
        status=1,
        speed_valid=True,
        stamp=10.0,
        motion_stamp=10.0,
        centroid_x=-0.4,
        centroid_y=0.3,
        velocity=np.array([0.0, -0.028, 0.0]),
        motion_covariance=np.diag([1e-6, 4e-6, 9e-6, 16e-6]),
        motion_process_noise=np.diag([1e-7, 2e-7]),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def contact(item=None, **changes):
    values = dict(now=10.1, horizon=2.0, offset=np.array([0.003, -0.004]),
                  offset_covariance=np.diag([1e-6, 2e-6]), timing_std=0.15)
    values.update(changes)
    return motion_at_contact(track() if item is None else item, **values)


def test_forecast_uses_continuous_acceleration_covariance():
    state = np.array([-0.4, 0.3, 0.002, -0.028])
    covariance = np.array([
        [4, 1, 2, 0], [1, 9, 0, -1], [2, 0, 16, 3], [0, -1, 3, 25],
    ], float) * 1e-6
    acceleration = np.array([[4, 1], [1, 9]], float) * 1e-7
    dt = 2.3
    transition = np.eye(4)
    transition[:2, 2:] = np.eye(2) * dt
    noise = np.block([
        [acceleration * dt**3 / 3, acceleration * dt**2 / 2],
        [acceleration * dt**2 / 2, acceleration * dt],
    ])
    predicted, propagated = forecast_motion(state, covariance, dt, acceleration)
    np.testing.assert_allclose(predicted, transition @ state)
    np.testing.assert_allclose(propagated, transition @ covariance @ transition.T + noise)
    np.testing.assert_allclose(propagated, propagated.T)
    assert np.linalg.eigvalsh(propagated).min() >= -1e-15


def test_process_noise_propagation_is_consistent_across_prediction_steps():
    state = np.array([-0.4, 0.3, 0.0, -0.028])
    covariance = track().motion_covariance
    acceleration = track().motion_process_noise
    halfway, halfway_covariance = forecast_motion(state, covariance, 0.7, acceleration)
    final, final_covariance = forecast_motion(halfway, halfway_covariance, 1.3, acceleration)
    direct, direct_covariance = forecast_motion(state, covariance, 2.0, acceleration)
    np.testing.assert_allclose(final, direct)
    np.testing.assert_allclose(final_covariance, direct_covariance, rtol=1e-12, atol=1e-15)


def test_zero_time_forecast_preserves_the_measurement_without_mutating_inputs():
    state = np.array([-0.4, 0.3, 0.0, -0.028])
    covariance = track().motion_covariance
    saved_state, saved_covariance = state.copy(), covariance.copy()
    predicted, propagated = forecast_motion(state, covariance, 0.0, track().motion_process_noise)
    np.testing.assert_array_equal(predicted, saved_state)
    np.testing.assert_array_equal(propagated, saved_covariance)
    np.testing.assert_array_equal(state, saved_state)
    np.testing.assert_array_equal(covariance, saved_covariance)


@pytest.mark.parametrize('state,dt', [
    (np.zeros(3), 1.0), (np.array([0.0, 0.0, np.nan, 0.0]), 1.0),
    (np.zeros(4), -0.1), (np.zeros(4), float('nan')), (np.zeros(4), float('inf')),
])
def test_invalid_forecast_state_or_time_rejected(state, dt):
    with pytest.raises(ValueError):
        forecast_motion(state, track().motion_covariance, dt, track().motion_process_noise)


def test_position_uncertainty_grows_with_contact_horizon():
    state = np.array([-0.4, 0.3, 0.0, -0.028])
    traces = [np.trace(forecast_motion(state, track().motion_covariance, dt,
                                      track().motion_process_noise)[1][:2, :2])
              for dt in [0.0, 0.5, 1.0, 3.0]]
    assert all(later > earlier for earlier, later in zip(traces, traces[1:]))


def test_prediction_accounts_for_age_offset_and_contact_timing_uncertainty():
    item = track()
    result = contact(item)
    velocity = item.velocity[:2]
    _, propagated = forecast_motion(
        np.r_[item.centroid_x, item.centroid_y, velocity], item.motion_covariance,
        2.1, item.motion_process_noise)
    expected_covariance = propagated[:2, :2] + np.diag([1e-6, 2e-6])
    expected_covariance += np.outer(velocity, velocity) * 0.15**2
    np.testing.assert_allclose(result['position'], [-0.397, 0.3 - 0.028 * 2.1 - 0.004])
    np.testing.assert_allclose(result['covariance'], expected_covariance)
    np.testing.assert_allclose(result['velocity'], velocity)
    assert result['age'] == pytest.approx(0.1)


def test_state_reference_time_prevents_double_counting_prediction_since_observation():
    # A lost-track Kalman state may already be predicted beyond its last camera frame.
    item = track(status=2, stamp=10.0, motion_stamp=10.08,
                 centroid_y=0.3 - 0.028 * 0.08)
    result = contact(item, allow_lost=True)
    np.testing.assert_allclose(result['position'], [-0.397, 0.3 - 0.028 * 2.1 - 0.004])
    assert result['age'] == pytest.approx(0.1)


@pytest.mark.parametrize('changes', [
    dict(stamp=9.7), dict(motion_stamp=9.7),
    dict(stamp=10.11), dict(motion_stamp=10.11),
    dict(stamp=float('nan')), dict(motion_stamp=float('inf')),
])
def test_stale_or_future_timestamps_block_trigger(changes):
    with pytest.raises(ValueError):
        contact(track(**changes))


@pytest.mark.parametrize('changes', [
    dict(status=0), dict(status=2), dict(status=3), dict(speed_valid=False),
])
def test_unconfirmed_lost_and_invalid_velocity_cannot_start_motion(changes):
    with pytest.raises(ValueError):
        contact(track(**changes))


def test_short_loss_is_only_allowed_after_motion_has_started():
    contact(track(status=2), allow_lost=True, now=10.149)
    with pytest.raises(ValueError):
        contact(track(status=2), allow_lost=True, now=10.151)
    with pytest.raises(ValueError):
        contact(track(status=2, speed_valid=False), allow_lost=True)


@pytest.mark.parametrize('matrix', [
    np.diag([1e-6, -1e-6, 1e-6, 1e-6]),
    np.array([[1e-6, 1e-6, 0, 0], [0, 1e-6, 0, 0],
              [0, 0, 1e-6, 0], [0, 0, 0, 1e-6]]),
    np.full((4, 4), np.nan), np.eye(3),
])
def test_invalid_motion_covariance_blocks_prediction(matrix):
    with pytest.raises(ValueError):
        contact(track(motion_covariance=matrix))


@pytest.mark.parametrize('matrix', [
    np.diag([1e-6, -1e-6]), np.array([[1e-6, 1e-6], [0, 1e-6]]),
    np.full((2, 2), np.inf), np.eye(3),
])
def test_invalid_process_and_offset_covariances_block_prediction(matrix):
    with pytest.raises(ValueError):
        contact(track(motion_process_noise=matrix))
    with pytest.raises(ValueError):
        contact(offset_covariance=matrix)


@pytest.mark.parametrize('changes', [
    dict(now=float('nan')), dict(horizon=-0.1), dict(horizon=float('inf')),
    dict(timing_std=-0.1), dict(timing_std=float('nan')),
    dict(offset=np.array([0.0, np.nan])),
])
def test_nonphysical_contact_inputs_rejected(changes):
    with pytest.raises(ValueError):
        contact(**changes)


def test_capture_bounds_project_correlated_covariance_onto_belt_axes():
    direction = np.array([3.0, -4.0])
    unit_direction = direction / 5.0
    cross_direction = np.array([0.8, 0.6])
    error = np.array([0.003, -0.004])
    covariance = np.array([[9e-6, 2e-6], [2e-6, 4e-6]])
    result = capture_bounds(error, covariance, [0.0, 0.0], direction)
    assert result['along_error'] == pytest.approx(error @ unit_direction)
    assert abs(result['cross_error']) == pytest.approx(abs(error @ cross_direction))
    assert result['along_bound'] == pytest.approx(
        abs(error @ unit_direction) + 3 * np.sqrt(unit_direction @ covariance @ unit_direction))
    assert result['cross_bound'] == pytest.approx(
        abs(error @ cross_direction) + 3 * np.sqrt(cross_direction @ covariance @ cross_direction))


def test_mean_at_goal_does_not_hide_unsafe_position_uncertainty():
    safe = capture_bounds([0.0, 0.0], np.eye(2) * 1e-6, [0.0, 0.0], [0.0, -1.0])
    unsafe = capture_bounds([0.0, 0.0], np.eye(2) * 1e-4, [0.0, 0.0], [0.0, -1.0])
    assert safe['along_error'] == unsafe['along_error'] == 0
    assert safe['cross_bound'] < 0.008
    assert unsafe['cross_bound'] > 0.008


@pytest.mark.parametrize('changes', [
    dict(direction=[0.0, 0.0]), dict(direction=[0.0, float('nan')]),
    dict(sigma=-1.0), dict(sigma=float('inf')),
    dict(covariance=np.diag([1e-6, -1e-6])),
    dict(position=[float('inf'), 0.0]),
])
def test_invalid_capture_inputs_rejected(changes):
    values = dict(position=[0.0, 0.0], covariance=np.eye(2) * 1e-6,
                  goal=[0.0, 0.0], direction=[0.0, -1.0], sigma=3.0)
    values.update(changes)
    with pytest.raises(ValueError):
        capture_bounds(**values)


def test_stage_only_starts_inside_feasible_time_window():
    timings = dict(approach=2.0, descend=3.5, contact=0.8, margin=0.4)
    assert stage_window(6.701, **timings) == 'WAIT'
    assert stage_window(6.7, **timings) == 'START'
    assert stage_window(6.5, **timings) == 'START'
    assert stage_window(6.3, **timings) == 'START'
    assert stage_window(6.299, **timings) == 'LATE'


@pytest.mark.parametrize('name', ['remaining', 'approach', 'descend', 'contact', 'margin'])
@pytest.mark.parametrize('bad_value', [-0.1, float('nan'), float('inf')])
def test_invalid_stage_timing_rejected(name, bad_value):
    values = dict(remaining=6.5, approach=2.0, descend=3.5, contact=0.8, margin=0.4)
    values[name] = bad_value
    with pytest.raises(ValueError):
        stage_window(**values)
