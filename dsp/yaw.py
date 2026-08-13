"""
Yaw 差模 PD 控制。
"""

KP = 8.0
KD = 1.0


def update(alpha, alpha_rate):
    return KP * alpha + KD * alpha_rate
