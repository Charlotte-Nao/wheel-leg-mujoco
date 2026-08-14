"""
左右虚拟腿长度 PID 控制。
"""

import numpy as np


WHEEL_TRACK = 0.5

KP = 1000.0
KD = 100.0
KI = 0.0
FF = 102.5
INTEGRAL_LIMIT = 30.0


class LegPID:
    def __init__(self):
        self.left_integral = 0.0
        self.right_integral = 0.0

    def reset(self):
        self.left_integral = 0.0
        self.right_integral = 0.0

    def _update(
        self,
        length,
        length_rate,
        length_ref,
        length_rate_ref,
        integral,
        dt,
    ):
        error = length_ref - length
        rate_error = length_rate_ref - length_rate
        integral += KI * error * dt
        integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, integral))

        force = KP * error + KD * rate_error + integral + FF

        return float(force), float(integral)

    def update(
        self,
        roll_error,
        left_length,
        right_length,
        left_length_rate,
        right_length_rate,
        length_ref,
        length_rate_ref=0.0,
        dt=0.001,
    ):
        delta_length = float(WHEEL_TRACK * np.sin(roll_error))
        current_length_difference = right_length - left_length
        target_length_difference = current_length_difference + delta_length
        left_length_ref = length_ref - target_length_difference / 2.0
        right_length_ref = length_ref + target_length_difference / 2.0

        left_force, self.left_integral = self._update(
            left_length,
            left_length_rate,
            left_length_ref,
            length_rate_ref,
            self.left_integral,
            dt,
        )
        right_force, self.right_integral = self._update(
            right_length,
            right_length_rate,
            right_length_ref,
            length_rate_ref,
            self.right_integral,
            dt,
        )

        return left_force, right_force, delta_length
