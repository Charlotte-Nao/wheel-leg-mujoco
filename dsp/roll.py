"""
Roll 差模支撑力补偿。
"""

FORCE_KP = 3000.0


def update(gamma):
    return -FORCE_KP * gamma
