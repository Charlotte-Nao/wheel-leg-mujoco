"""
虚拟腿长度控制。
完全使用参考 Simulink 参数：L_ref=0.2 m，Kp=150，Ki=0，Kd=0。
"""

L_REF = 0.2
KP = 150.0
KD = 0.0


def update(length, length_rate=0.0):
    return KP * (L_REF - length) - KD * length_rate