"""Read XYZ from a PointCloud2 as an Nx3 float32 array."""
import numpy as np
from sensor_msgs.msg import PointCloud2

_DTYPES = {7: np.float32, 8: np.float64}  # PointField FLOAT32/FLOAT64


def read_points_xyz(cloud: PointCloud2):
    if cloud is None or cloud.height * cloud.width == 0:
        return np.zeros((0, 3), np.float32)
    fields = {f.name: f for f in cloud.fields}
    if cloud.row_step < cloud.width * cloud.point_step or len(cloud.data) < cloud.row_step * cloud.height:
        raise ValueError('Invalid PointCloud2 strides/data size')
    out = np.zeros((cloud.height * cloud.width, 3), np.float32)
    for i, name in enumerate(('x', 'y', 'z')):
        f = fields.get(name)
        if f is None or f.datatype not in _DTYPES or f.count != 1:
            raise ValueError('XYZ fields must each be one FLOAT32/FLOAT64 value')
        dt = np.dtype(_DTYPES[f.datatype]).newbyteorder('>' if cloud.is_bigendian else '<')
        if f.offset < 0 or f.offset + dt.itemsize > cloud.point_step:
            raise ValueError('PointCloud2 field exceeds point_step')
        arr = np.ndarray(shape=(cloud.height, cloud.width), dtype=dt, buffer=cloud.data,
                         offset=f.offset, strides=(cloud.row_step, cloud.point_step))
        out[:, i] = arr.ravel().astype(np.float32)
    return out
