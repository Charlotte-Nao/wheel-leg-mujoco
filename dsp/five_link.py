"""
五连杆运动学。
按照 MATLAB VMC.m 的几何和 +sqrt 装配分支计算 L、phi0、Jacobian 和速度。
"""

import numpy as np

L1 = 0.21
L2 = 0.25
L3 = 0.25
L4 = 0.21
L_AE = 0.0


def _geometry(q1, q4):
    xb, yb = L1 * np.cos(q1), L1 * np.sin(q1)
    xd, yd = L_AE + L4 * np.cos(q4), L4 * np.sin(q4)

    A0 = 2.0 * L2 * (xd - xb)
    B0 = 2.0 * L2 * (yd - yb)
    C0 = L2**2 + (xd - xb)**2 + (yd - yb)**2 - L3**2

    disc = A0**2 + B0**2 - C0**2
    if disc < -1e-12:
        raise ValueError(f"Five-link configuration unreachable: discriminant={disc}")
    disc = max(disc, 0.0)

    numerator = B0 + np.sqrt(disc)
    denominator = A0 + C0
    if abs(denominator) < 1e-12:
        ratio = np.copysign(np.inf, numerator)
    else:
        ratio = numerator / denominator

    phi2 = 2.0 * np.arctan(ratio)
    xc = xb + L2 * np.cos(phi2)
    yc = yb + L2 * np.sin(phi2)
    phi3 = np.arctan2(yc - yd, xc - xd)

    x0 = xc - L_AE / 2.0
    length = np.hypot(x0, yc)
    phi0 = np.arctan2(yc, x0)

    return xb, yb, xd, yd, xc, yc, phi2, phi3, length, phi0


def position(q1, q4):
    *_, length, phi0 = _geometry(q1, q4)
    return np.array([length, phi0])


def jacobian(q1, q4):
    xb, yb, xd, yd, xc, yc, phi2, phi3, length, _ = _geometry(q1, q4)

    n1 = np.array([-np.sin(q1), np.cos(q1)])
    n2 = np.array([-np.sin(phi2), np.cos(phi2)])
    n3 = np.array([-np.sin(phi3), np.cos(phi3)])
    n4 = np.array([-np.sin(q4), np.cos(q4)])

    M = np.column_stack((L2 * n2, -L3 * n3))
    rhs = np.column_stack((-L1 * n1, L4 * n4))
    angle_derivative = np.linalg.solve(M, rhs)
    dphi2 = angle_derivative[0]

    dC_dq1 = L1 * n1 + L2 * n2 * dphi2[0]
    dC_dq4 = L2 * n2 * dphi2[1]
    dC = np.column_stack((dC_dq1, dC_dq4))

    x0 = xc - L_AE / 2.0
    J_L = np.array([x0, yc]) @ dC / length
    J_phi = np.array([-yc, x0]) @ dC / length**2

    return np.vstack((J_L, J_phi))


def speed(dq1, dq4, q1, q4):
    return jacobian(q1, q4) @ np.array([dq1, dq4])


def virtual_state(q1, q4, dq1, dq4):
    length, phi0 = position(q1, q4)
    length_rate, phi0_rate = speed(dq1, dq4, q1, q4)

    theta = np.pi / 2.0 - phi0
    theta_rate = -phi0_rate

    return length, theta, length_rate, theta_rate