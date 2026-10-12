"""Optional GUI startup and closure must not disrupt motion control."""
import subprocess
from unittest.mock import Mock, patch

from ament_index_python.packages import PackageNotFoundError
from nero7_control.visualization import SpeedImageViewer


def test_no_desktop_skips_gui_without_launching_a_process():
    with patch.dict('os.environ', {}, clear=True), \
            patch('nero7_control.visualization.subprocess.Popen') as popen:
        viewer = SpeedImageViewer(Mock(), '/perception/debug_image')
        viewer.check()
        viewer.close()
    popen.assert_not_called()


def test_viewer_uses_configured_simulation_topic_and_closes_its_process():
    process = Mock()
    process.poll.return_value = None
    with patch.dict('os.environ', {'DISPLAY': ':1'}), \
            patch('nero7_control.visualization.get_package_prefix', return_value='/opt/ros/humble'), \
            patch('nero7_control.visualization.subprocess.Popen', return_value=process) as popen:
        viewer = SpeedImageViewer(Mock(), '/sim/perception/debug_image')
    assert popen.call_args.args[0] == [
        '/opt/ros/humble/lib/rqt_image_view/rqt_image_view', '/sim/perception/debug_image']
    viewer.close()
    viewer.close()
    process.terminate.assert_called_once()
    process.kill.assert_not_called()


def test_unresponsive_viewer_is_killed_after_bounded_shutdown_wait():
    viewer = SpeedImageViewer.__new__(SpeedImageViewer)
    viewer._process = process = Mock()
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired('viewer', .5), 0]
    viewer.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()


def test_missing_gui_dependency_is_reported_without_raising():
    logger = Mock()
    with patch.dict('os.environ', {'DISPLAY': ':1'}), \
            patch('nero7_control.visualization.get_package_prefix',
                  side_effect=PackageNotFoundError('rqt_image_view')):
        viewer = SpeedImageViewer(logger, '/perception/debug_image')
    viewer.close()
    logger.warning.assert_called_once()


def test_closed_or_failed_window_is_not_reopened():
    viewer = SpeedImageViewer.__new__(SpeedImageViewer)
    viewer._logger = Mock()
    viewer._process = process = Mock()
    process.poll.return_value = 1
    process.returncode = 1
    viewer.check()
    viewer.check()
    viewer.close()
    viewer._logger.warning.assert_called_once()
    process.terminate.assert_not_called()
