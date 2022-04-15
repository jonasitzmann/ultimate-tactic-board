import numpy as np
import cfg


def get_hex_angle(pos, dist=10, min_sideline_dist=None):
    angle = 0
    sin_60 = np.sqrt(3) / 2
    x_0 = sin_60 * dist
    if min_sideline_dist is None:
        min_sideline_dist = 0.5 * dist * 1.1
    is_positive = pos[0] > cfg.field_width_m / 2
    sign = 1 if is_positive else -1
    space_available = (cfg.field_width_m - pos[0] if is_positive else pos[0]) - min_sideline_dist
    if space_available < x_0:
        angle = np.rad2deg(np.arcsin(space_available / dist)) - 60
        angle *= -sign
    angle = np.clip(angle, -90, 90)
    return angle
