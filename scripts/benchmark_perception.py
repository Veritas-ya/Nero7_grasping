"""Synthetic 1280x720 legacy-plane fitting comparison; no ROS or hardware."""
import importlib.util
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from nero7_perception.conveyor_model import BeltPlane

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('perception_baseline',
    root/'docs/perception_baseline/conveyor_model.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)

rng = np.random.default_rng(827)
v, u = np.indices((720, 1280))
truth = .9+.0001*u-.00008*v
depth = truth+rng.normal(0., .001, truth.shape)
depth[250:400, 400:650] -= .04
depth[rng.random(depth.shape) < .01] = np.nan
report = dict(scope='synthetic plane fitting only; excludes ROS, network and real camera',
              image_size=[1280, 720], fits={})
for label, constructor in [('before', baseline.BeltPlane), ('after', BeltPlane)]:
    model = constructor()
    times = []
    for _ in range(5):
        start = perf_counter()
        if not model.fit(depth):
            raise RuntimeError('Synthetic plane fitting failed')
        times.append((perf_counter()-start)*1000)
    error = float(np.max(np.abs(model.plane_z(u, v)-truth)))
    if error > .006:
        raise RuntimeError('Synthetic plane error exceeds fitting threshold')
    report['fits'][label] = dict(median_ms=float(np.median(times)),
                                 max_plane_error_m=error, sample_limit=getattr(model, 'max_points', 30000))
report['median_speedup'] = report['fits']['before']['median_ms']/report['fits']['after']['median_ms']
print(json.dumps(report, indent=2))
