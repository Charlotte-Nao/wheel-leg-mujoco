"""
Roll 差模腿长与支撑力控制。
"""

import numpy as np


WHEEL_TRACK = 0.5

LENGTH_KP = 1000.0
LENGTH_KD = 100.0
FORCE_KP = 100.0


def update(gamma, left_length, right_length, left_length_rate, right_length_rate):
    delta_length = WHEEL_TRACK * np.sin(gamma)

    length_error = ( -delta_length - (left_length - right_length) )
    length_rate_error = left_length_rate - right_length_rate

    delta_F_length = (
        LENGTH_KP * length_error / 2.0
        - LENGTH_KD * length_rate_error / 2.0
    )
    delta_F_roll = -FORCE_KP * gamma

    return delta_length, delta_F_length + delta_F_roll
