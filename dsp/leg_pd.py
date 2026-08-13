"""
虚拟腿长度 PID 控制。
"""

L_REF = 0.2
KP = 1000.0
KD = 100.0
KI = 100.0
FF = 102.5
INTEGRAL_LIMIT = 30.0


class LegPID:
    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, length, length_rate=0.0, dt=0.001):
        error = L_REF - length
        self.integral += KI * error * dt
        self.integral = max(-INTEGRAL_LIMIT, min(INTEGRAL_LIMIT, self.integral))

        return KP * error - KD * length_rate + self.integral + FF
