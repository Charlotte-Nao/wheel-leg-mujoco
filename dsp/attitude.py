"""
车体姿态与欧拉角速度解算。
"""

import numpy as np


def update(quaternion, angular_velocity):
    w, x, y, z = quaternion

    gamma = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x**2 + y**2),
    )
    phi = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    alpha = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y**2 + z**2),
    )

    omega_x, omega_y, omega_z = angular_velocity
    cos_phi = np.cos(phi)

    gamma_rate = (
        omega_x
        + np.sin(gamma) * np.tan(phi) * omega_y
        + np.cos(gamma) * np.tan(phi) * omega_z
    )
    phi_rate = np.cos(gamma) * omega_y - np.sin(gamma) * omega_z
    alpha_rate = (
        np.sin(gamma) / cos_phi * omega_y
        + np.cos(gamma) / cos_phi * omega_z
    )

    return gamma, phi, alpha, gamma_rate, phi_rate, alpha_rate
