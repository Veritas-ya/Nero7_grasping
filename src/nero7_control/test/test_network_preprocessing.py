import numpy as np
import pytest

from nero7_control.grasp_inference import prepare_network_input


def test_rgb_keeps_channel_relationships_with_training_global_mean():
    rgb = np.zeros((16, 16, 3), np.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = 128
    result = prepare_network_input(rgb, np.full((16, 16), 1000, np.uint16))
    expected = np.array([1., 128/255., 0.], np.float32)
    np.testing.assert_allclose(result[1:, 8, 8], expected-expected.mean(), atol=1e-6)
    assert result.shape == (4, 16, 16)
    assert result.dtype == np.float32


def test_depth_uses_metres_and_training_clipping():
    depth = np.full((16, 16), 1000, np.uint16)
    depth[:, 8:] = 1100
    result = prepare_network_input(np.zeros((16, 16, 3), np.uint8), depth)
    np.testing.assert_allclose(result[0, 8, [0, 15]], [-.05, .05], atol=1e-6)


def test_inpainting_does_not_modify_measured_depth():
    depth = np.full((32, 32), 900, np.uint16)
    depth[14:18, 14:18] = 0
    original = depth.copy()
    result = prepare_network_input(np.zeros((32, 32, 3), np.uint8), depth)
    assert np.isfinite(result).all()
    assert abs(result[0, 16, 16]) < .01
    np.testing.assert_array_equal(depth, original)


def test_nonfinite_depth_is_filled_only_in_network_input():
    depth = np.full((16, 16), 1000., np.float32)
    depth[8, 8] = np.nan
    depth[9, 8] = np.inf
    result = prepare_network_input(np.zeros((16, 16, 3), np.uint8), depth)
    assert np.isfinite(result).all()
    assert np.isnan(depth[8, 8]) and np.isinf(depth[9, 8])


def test_all_missing_depth_is_rejected():
    with pytest.raises(ValueError, match='No valid depth'):
        prepare_network_input(np.zeros((16, 16, 3), np.uint8), np.zeros((16, 16), np.uint16))
