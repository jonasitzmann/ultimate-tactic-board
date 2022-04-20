import numpy as np
import cfg
import constants
from functools import lru_cache


@lru_cache(maxsize=100)
def get_angle(space_available, dist, sign):
    angle = np.rad2deg(np.arcsin(space_available / dist)) - 60
    if space_available < 0:
        angle = -90
    return angle * -sign


def get_hex_angle(pos, dist=None, min_sideline_dist=None):
    dist = cfg.hex_dist_m if dist is None else dist
    angle = 0
    x_0 = constants.sin_60 * dist
    if min_sideline_dist is None:
        min_sideline_dist = 0.5 * dist * 1.1
    is_positive = pos[0] > cfg.field_width_m / 2
    sign = 1 if is_positive else -1
    space_available = (cfg.field_width_m - pos[0] if is_positive else pos[0]) - min_sideline_dist
    if space_available < x_0:
        angle = get_angle(space_available, dist, sign)
    angle = np.clip(angle, -90, 90)
    return angle
