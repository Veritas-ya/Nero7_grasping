"""Read XYZ from a PointCloud2 as an Nx3 float32 array."""
import numpy as np
from sensor_msgs.msg import PointCloud2

_DTYPES = {1: np.uint8, 2: np.uint16, 4: np.float32, 7: np.float64}


def read_points_xyz(cloud: PointCloud2):
    if cloud is None or cloud.height * cloud.width == 0:
        return np.zeros((0, 3), np.float32)
    fields = {f.name: f for f in cloud.fields}
    n = cloud.height * cloud.width
    out = np.zeros((n, 3), np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        f = fields[name]
        dt = _DTYPES[f.datatype]
        # interleaved field access via strides
        arr = np.ndarray(shape=(n,), dtype=dt, buffer=cloud.data,
                         offset=f.offset, strides=(cloud.point_step,))
        out[:, i] = arr.astype(np.float32)
    return out
