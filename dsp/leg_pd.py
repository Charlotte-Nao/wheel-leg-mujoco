"""
虚拟腿长度控制。
"""

L_REF = 0.2
KP = 1000.0
KD = 100.0
FF = 102.5


def update(length, length_rate=0.0):
    return KP * (L_REF - length) - KD * length_rate + FF
