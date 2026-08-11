"""
VMC虚功映射。
按照 MATLAB VMC.m：tau = J.T @ [F, Tp]，输出两个髋关节力矩。
"""

import numpy as np

from dsp.five_link import jacobian


def update(F, Tp, q1, q4):
    J = jacobian(q1, q4)
    return J.T @ np.array([F, Tp])