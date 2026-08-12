"""
轮腿机器人离散 LQR。
直接按当前 Python/MuJoCo 坐标建立动力学模型并在线按腿长计算增益。
状态：[theta, dtheta, x, dx, phi, dphi]
输入：[T, Tp]
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import expm, solve_discrete_are


DT = 0.001

L_MIN = 0.04
L_MAX = 0.46
GAIN_STEP = 0.005

R_WHEEL = 0.06
M_WHEEL = 0.6
M_LEG = 0.95
M_BODY = 18.9

I_WHEEL = 0.5 * M_WHEEL * R_WHEEL**2
I_LEG = 0.042737089
I_BODY = 0.589763361

BODY_COM_OFFSET = 0.0
G = 9.81

Q = np.diag([
    5000.0,
    200.0,
    2.0,
    1.0,
    5000.0,
    20.0,
])

R = np.diag([
    50.0,
    8.0,
])


def _dynamics(state, control, leg_length):
    theta, dtheta, _, dx, phi, dphi = state
    T, Tp = control

    L = leg_length / 2.0
    Lm = leg_length / 2.0
    l = BODY_COM_OFFSET

    def residual(acc):
        ddtheta, ddx, ddphi = acc

        Nm = M_BODY * (
            ddx
            - (L + Lm) * ddtheta * np.cos(theta)
            + (L + Lm) * dtheta**2 * np.sin(theta)
            + l * (
                ddphi * np.cos(phi)
                - dphi**2 * np.sin(phi)
            )
        )

        Pm = M_BODY * G + M_BODY * (
            -(L + Lm) * ddtheta * np.sin(theta)
            - (L + Lm) * dtheta**2 * np.cos(theta)
            - l * ddphi * np.sin(phi)
            - l * dphi**2 * np.cos(phi)
        )

        N = Nm + M_LEG * (
            ddx
            - L * ddtheta * np.cos(theta)
            + L * dtheta**2 * np.sin(theta)
        )

        P = Pm + M_LEG * G + M_LEG * (
            -L * dtheta**2 * np.cos(theta)
            - L * ddtheta * np.sin(theta)
        )

        eq1 = ddx - (
            T - N * R_WHEEL
        ) / (
            I_WHEEL / R_WHEEL
            + M_WHEEL * R_WHEEL
        )

        eq2 = (
            (P * L + Pm * Lm) * np.sin(theta)
            + (N * L + Nm * Lm) * np.cos(theta)
            + T
            + Tp
            - I_LEG * ddtheta
        )

        eq3 = (
            -Tp
            + Nm * l * np.cos(phi)
            - Pm * l * np.sin(phi)
            + I_BODY * ddphi
        )

        return np.array([eq1, eq2, eq3], dtype=float)

    zero = np.zeros(3)
    r0 = residual(zero)

    E = np.column_stack((
        residual(np.array([1.0, 0.0, 0.0])) - r0,
        residual(np.array([0.0, 1.0, 0.0])) - r0,
        residual(np.array([0.0, 0.0, 1.0])) - r0,
    ))

    ddtheta, ddx, ddphi = np.linalg.solve(E, -r0)

    return np.array([
        dtheta,
        ddtheta,
        dx,
        ddx,
        dphi,
        ddphi,
    ])


def _linearize(leg_length):
    state0 = np.zeros(6)
    control0 = np.zeros(2)

    A = np.zeros((6, 6))
    B = np.zeros((6, 2))

    state_eps = np.array([
        1e-6,
        1e-6,
        1e-5,
        1e-5,
        1e-6,
        1e-6,
    ])

    control_eps = np.array([
        1e-5,
        1e-5,
    ])

    for i in range(6):
        eps = state_eps[i]

        plus = state0.copy()
        minus = state0.copy()

        plus[i] += eps
        minus[i] -= eps

        A[:, i] = (
            _dynamics(plus, control0, leg_length)
            - _dynamics(minus, control0, leg_length)
        ) / (2.0 * eps)

    for i in range(2):
        eps = control_eps[i]

        plus = control0.copy()
        minus = control0.copy()

        plus[i] += eps
        minus[i] -= eps

        B[:, i] = (
            _dynamics(state0, plus, leg_length)
            - _dynamics(state0, minus, leg_length)
        ) / (2.0 * eps)

    return A, B


def _discretize(A, B):
    n = A.shape[0]
    m = B.shape[1]

    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n] = A
    augmented[:n, n:] = B

    discrete = expm(augmented * DT)

    return discrete[:n, :n], discrete[:n, n:]


@lru_cache(maxsize=128)
def _gain_at_length(leg_length):
    A, B = _linearize(float(leg_length))
    Ad, Bd = _discretize(A, B)

    P = solve_discrete_are(Ad, Bd, Q, R)

    return np.linalg.solve(
        R + Bd.T @ P @ Bd,
        Bd.T @ P @ Ad,
    )


class LQR:
    def __init__(self):
        pass

    @staticmethod
    def gain(L):
        L = float(np.clip(L, L_MIN, L_MAX))

        index = (L - L_MIN) / GAIN_STEP
        low_index = int(np.floor(index))
        high_index = int(np.ceil(index))

        L_low = L_MIN + low_index * GAIN_STEP
        L_high = L_MIN + high_index * GAIN_STEP

        L_low = round(float(np.clip(L_low, L_MIN, L_MAX)), 6)
        L_high = round(float(np.clip(L_high, L_MIN, L_MAX)), 6)

        K_low = _gain_at_length(L_low)

        if L_low == L_high:
            return K_low.copy()

        K_high = _gain_at_length(L_high)
        alpha = (L - L_low) / (L_high - L_low)

        return (1.0 - alpha) * K_low + alpha * K_high

    def update(self, L, state):
        state = np.asarray(state, dtype=float).reshape(6)

        K = self.gain(L)
        T, Tp = -K @ state

        return float(T), float(Tp)