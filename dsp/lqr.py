"""
LQR增益调度。
直接读取 MATLAB 生成的 reference_lqr_build.mat，在线根据当前腿长 L 计算 K(L)，不在 Python 中重新计算 ABK、dlqr 或 polyfit。
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


class LQR:
    def __init__(self, mat_path=None):
        if mat_path is None:
            mat_path = Path(__file__).with_name("reference_lqr_build.mat")

        data = loadmat(mat_path)
        self.coeff = np.asarray(data["fit_coeff"], dtype=float)

        if self.coeff.shape != (2, 6, 4):
            raise ValueError(f"fit_coeff shape should be (2, 6, 4), got {self.coeff.shape}")

    def gain(self, length):
        c = self.coeff
        return ((c[:, :, 0] * length + c[:, :, 1]) * length + c[:, :, 2]) * length + c[:, :, 3]

    def update(self, length, state):
        state = np.asarray(state, dtype=float).reshape(6)
        return -self.gain(length) @ state