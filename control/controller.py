"""Platform-independent wheel-legged robot control core."""

import numpy as np

from control.targets import TargetStore
from control.types import ActuatorCommand, ControlTelemetry, RobotState
from dsp.leg_pd import LegPID
from dsp.lqr import LQR
from dsp.roll import update as roll
from dsp.theta_pd import update as theta_pd
from dsp.vmc import update as vmc
from dsp.yaw import update as yaw


WHEEL_TORQUE_LIMIT = 10.0
HIP_TORQUE_LIMIT = 40.0


def angle_error(angle, target):
    """Return the shortest signed angular displacement from target to angle."""

    difference = angle - target
    return float(np.arctan2(np.sin(difference), np.cos(difference)))


class Controller:
    def __init__(self, targets=None):
        self.targets = targets if targets is not None else TargetStore()
        if not isinstance(self.targets, TargetStore):
            raise TypeError("targets must be a TargetStore instance")

        self.lqr = LQR()
        self.leg_pid = LegPID()

    def reset(self):
        self.leg_pid.reset()

    def update(self, state, dt):
        if not isinstance(state, RobotState):
            raise TypeError("state must be a RobotState instance")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        target = self.targets.snapshot()

        lqr_error = (
            np.asarray(state.lqr_state)
            - np.asarray(target.lqr_state)
        )

        common_wheel_torque, common_leg_torque = self.lqr.update(
            state.length,
            lqr_error,
        )

        roll_position_error = state.roll - target.roll
        left_force, right_force, delta_length = self.leg_pid.update(
            roll_error=roll_position_error,
            left_length=state.left.length,
            right_length=state.right.length,
            left_length_rate=state.left.length_rate,
            right_length_rate=state.right.length_rate,
            length_ref=target.leg_length,
            length_rate_ref=target.leg_length_rate,
            dt=dt,
        )
        common_force = (left_force + right_force) / 2.0

        yaw_position_error = angle_error(state.yaw, target.yaw)
        yaw_velocity_error = state.yaw_rate - target.yaw_rate
        theta_difference_error = (
            state.left.theta
            - state.right.theta
            - target.theta_difference
        )
        theta_difference_rate_error = (
            state.left.theta_rate
            - state.right.theta_rate
            - target.theta_difference_rate
        )

        delta_wheel_torque = yaw(
            yaw_position_error,
            yaw_velocity_error,
        )
        delta_leg_torque = theta_pd(
            theta_difference_error,
            theta_difference_rate_error,
        )
        delta_force = roll(roll_position_error)

        left_wheel_torque = common_wheel_torque + delta_wheel_torque
        right_wheel_torque = common_wheel_torque - delta_wheel_torque
        left_leg_torque = common_leg_torque + delta_leg_torque
        right_leg_torque = common_leg_torque - delta_leg_torque
        left_force += delta_force
        right_force -= delta_force

        left_q1_torque, left_q4_torque = vmc(
            left_force,
            left_leg_torque,
            state.left.q1,
            state.left.q4,
        )
        right_q1_torque, right_q4_torque = vmc(
            right_force,
            right_leg_torque,
            state.right.q1,
            state.right.q4,
        )

        left_wheel_torque = float(np.clip(
            left_wheel_torque,
            -WHEEL_TORQUE_LIMIT,
            WHEEL_TORQUE_LIMIT,
        ))
        right_wheel_torque = float(np.clip(
            right_wheel_torque,
            -WHEEL_TORQUE_LIMIT,
            WHEEL_TORQUE_LIMIT,
        ))
        left_q1_torque = float(np.clip(
            left_q1_torque,
            -HIP_TORQUE_LIMIT,
            HIP_TORQUE_LIMIT,
        ))
        left_q4_torque = float(np.clip(
            left_q4_torque,
            -HIP_TORQUE_LIMIT,
            HIP_TORQUE_LIMIT,
        ))
        right_q1_torque = float(np.clip(
            right_q1_torque,
            -HIP_TORQUE_LIMIT,
            HIP_TORQUE_LIMIT,
        ))
        right_q4_torque = float(np.clip(
            right_q4_torque,
            -HIP_TORQUE_LIMIT,
            HIP_TORQUE_LIMIT,
        ))

        command = ActuatorCommand(
            left_wheel=left_wheel_torque,
            left_q1=left_q1_torque,
            left_q4=left_q4_torque,
            right_wheel=right_wheel_torque,
            right_q1=right_q1_torque,
            right_q4=right_q4_torque,
        )
        telemetry = ControlTelemetry(
            length=state.length,
            length_rate=state.length_rate,
            theta=state.theta,
            theta_rate=state.theta_rate,
            roll=state.roll,
            roll_rate=state.roll_rate,
            pitch=state.pitch,
            pitch_rate=state.pitch_rate,
            yaw=state.yaw,
            yaw_rate=state.yaw_rate,
            x=state.x,
            x_rate=state.x_rate,
            lqr_error=tuple(float(value) for value in lqr_error),
            yaw_error=yaw_position_error,
            yaw_rate_error=yaw_velocity_error,
            roll_error=roll_position_error,
            theta_difference_error=theta_difference_error,
            theta_difference_rate_error=theta_difference_rate_error,
            common_force=common_force,
            common_wheel_torque=common_wheel_torque,
            common_leg_torque=common_leg_torque,
            delta_length=delta_length,
            delta_force=delta_force,
            delta_wheel_torque=delta_wheel_torque,
            delta_leg_torque=delta_leg_torque,
            left_force=left_force,
            right_force=right_force,
            left_wheel_torque=left_wheel_torque,
            right_wheel_torque=right_wheel_torque,
            left_leg_torque=left_leg_torque,
            right_leg_torque=right_leg_torque,
            left_q1_torque=left_q1_torque,
            left_q4_torque=left_q4_torque,
            right_q1_torque=right_q1_torque,
            right_q4_torque=right_q4_torque,
        )

        return command, telemetry
