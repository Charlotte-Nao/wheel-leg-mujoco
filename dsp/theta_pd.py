"""
左右虚拟腿角度差模 PD 控制。
"""

KP = 20.0
KD = 2.0


def update(theta_error, theta_rate_error):
    return -KP * theta_error - KD * theta_rate_error
