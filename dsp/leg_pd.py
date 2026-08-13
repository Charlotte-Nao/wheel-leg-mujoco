"""
左右虚拟腿长度 PID 控制。
"""

import numpy as np


L_REF = 0.2
WHEEL_TRACK = 0.5

KP = 1000.0
KD = 100.0
KI = 100.0
FF = 102.5
INTEGRAL_LIMIT = 30.0


class LegPID:
    def __init__(self):
        self.left_integral = 0.0
        self.right_integral = 0.0

    def reset(self):
        self.left_integral = 0.0
        self.right_integral = 0.0

    def _update(self, length, length_rate, length_ref, integral, dt):
        error = length_ref - length
        integral += KI * error * dt
        integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, integral))

        force = KP * error - KD * length_rate + integral + FF

        return force, integral

    def update(
        self,
        gamma,
        left_length,
        right_length,
        left_length_rate,
        right_length_rate,
        dt=0.001,
    ):
        delta_length = WHEEL_TRACK * np.sin(gamma)
        left_length_ref = L_REF - delta_length / 2.0
        right_length_ref = L_REF + delta_length / 2.0

        left_force, self.left_integral = self._update(left_length,left_length_rate,left_length_ref,self.left_integral,dt,)
        right_force, self.right_integral = self._update(right_length,right_length_rate,right_length_ref,self.right_integral,dt,)

        return left_force, right_force, delta_length
