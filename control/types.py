"""Data contracts shared by the control core and platform adapters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LegState:
    """Measured state of one virtual leg.

    ``theta`` and ``theta_rate`` are expressed relative to the world vertical,
    after body pitch has been included.
    """

    q1: float
    q4: float
    length: float
    theta: float
    length_rate: float
    theta_rate: float


@dataclass(frozen=True, slots=True)
class RobotState:
    """Measured robot state consumed by the platform-independent controller.

    ``x`` and ``x_rate`` are forward-path distance and speed in the platform's
    sagittal direction, not position and velocity on the world X axis.
    """

    left: LegState
    right: LegState
    roll: float
    roll_rate: float
    pitch: float
    pitch_rate: float
    yaw: float
    yaw_rate: float
    x: float
    x_rate: float

    @property
    def length(self):
        return (self.left.length + self.right.length) / 2.0

    @property
    def length_rate(self):
        return (self.left.length_rate + self.right.length_rate) / 2.0

    @property
    def theta(self):
        return (self.left.theta + self.right.theta) / 2.0

    @property
    def theta_rate(self):
        return (self.left.theta_rate + self.right.theta_rate) / 2.0

    @property
    def lqr_state(self):
        return (
            self.theta,
            self.theta_rate,
            self.x,
            self.x_rate,
            self.pitch,
            self.pitch_rate,
        )


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """Final actuator torques after control allocation and saturation."""

    left_wheel: float
    left_q1: float
    left_q4: float
    right_wheel: float
    right_q1: float
    right_q4: float


@dataclass(frozen=True, slots=True)
class ControlTelemetry:
    """Control-cycle values intended for logging and diagnostics."""

    length: float
    length_rate: float
    theta: float
    theta_rate: float
    roll: float
    roll_rate: float
    pitch: float
    pitch_rate: float
    yaw: float
    yaw_rate: float
    x: float
    x_rate: float
    lqr_error: tuple[float, float, float, float, float, float]
    yaw_error: float
    yaw_rate_error: float
    roll_error: float
    theta_difference_error: float
    theta_difference_rate_error: float
    common_force: float
    common_wheel_torque: float
    common_leg_torque: float
    delta_length: float
    delta_force: float
    delta_wheel_torque: float
    delta_leg_torque: float
    left_force: float
    right_force: float
    left_wheel_torque: float
    right_wheel_torque: float
    left_leg_torque: float
    right_leg_torque: float
    left_q1_torque: float
    left_q4_torque: float
    right_q1_torque: float
    right_q4_torque: float
