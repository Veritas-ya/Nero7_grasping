"""Manage the optional image viewer outside the motion-control process."""
import os
from pathlib import Path
import subprocess

from ament_index_python.packages import get_package_prefix, PackageNotFoundError


class SpeedImageViewer:
    def __init__(self, logger, topic):
        self._logger = logger
        self._process = None
        if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            logger.info(f'No desktop display; speed image remains available on {topic}')
            return
        try:
            executable = Path(get_package_prefix('rqt_image_view')) / 'lib/rqt_image_view/rqt_image_view'
            # Execute the viewer directly so shutdown also closes the window.
            self._process = subprocess.Popen(
                [str(executable), topic],
                start_new_session=True)
        except (PackageNotFoundError, OSError) as error:
            logger.warning(f'Cannot open speed visualization: {error}; image topic: {topic}')

    def check(self):
        if self._process is not None and self._process.poll() is not None:
            code = self._process.returncode
            if code:
                self._logger.warning(f'Speed visualization exited with code {code}')
            self._process = None

    def close(self):
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=0.5)
