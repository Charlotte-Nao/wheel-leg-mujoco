"""MuJoCo state and actuator adapter for the control core."""

import math

import mujoco

from control.types import ActuatorCommand, LegState, RobotState
from dsp.attitude import update as attitude
from dsp.five_link import virtual_state


class MujocoAdapter:
    def __init__(self, model):
        self.model = model

        self.left_q1_qpos, self.left_q1_dof = self._joint("left_q1")
        self.left_q4_qpos, self.left_q4_dof = self._joint("left_q4")
        self.right_q1_qpos, self.right_q1_dof = self._joint("right_q1")
        self.right_q4_qpos, self.right_q4_dof = self._joint("right_q4")

        self.chassis = self._id(mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.chassis_gyro = self._sensor("chassis_gyro")
        self.left_site_c = self._id(
            mujoco.mjtObj.mjOBJ_SITE,
            "left_C_wheel",
        )
        self.right_site_c = self._id(
            mujoco.mjtObj.mjOBJ_SITE,
            "right_C_wheel",
        )

        self.left_act_q1 = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_hip_q1_motor",
        )
        self.left_act_q4 = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_hip_q4_motor",
        )
        self.left_act_wheel = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "left_wheel_motor",
        )
        self.right_act_q1 = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_hip_q1_motor",
        )
        self.right_act_q4 = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_hip_q4_motor",
        )
        self.right_act_wheel = self._id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            "right_wheel_motor",
        )

        self.forward_position = 0.0
        self.last_world_position = None
        self.last_yaw = None

    def _id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return obj_id

    def _joint(self, name):
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _sensor(self, name):
        sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return slice(address, address + dimension)

    def _raw_leg_state(self, data, q1_qpos, q1_dof, q4_qpos, q4_dof):
        q1 = float(data.qpos[q1_qpos])
        q4 = float(data.qpos[q4_qpos])
        dq1 = float(data.qvel[q1_dof])
        dq4 = float(data.qvel[q4_dof])
        length, theta, length_rate, theta_rate = virtual_state(
            q1,
            q4,
            dq1,
            dq4,
        )
        return (
            q1,
            q4,
            float(length),
            float(theta),
            float(length_rate),
            float(theta_rate),
        )

    def _world_position(self, data):
        left_x = float(data.site_xpos[self.left_site_c, 0])
        left_y = float(data.site_xpos[self.left_site_c, 1])
        right_x = float(data.site_xpos[self.right_site_c, 0])
        right_y = float(data.site_xpos[self.right_site_c, 1])
        return (
            (left_x + right_x) / 2.0,
            (left_y + right_y) / 2.0,
        )

    def reset(self, data):
        self.forward_position = 0.0
        self.last_world_position = self._world_position(data)
        self.last_yaw = None

    def read_state(self, data):
        left = self._raw_leg_state(
            data,
            self.left_q1_qpos,
            self.left_q1_dof,
            self.left_q4_qpos,
            self.left_q4_dof,
        )
        right = self._raw_leg_state(
            data,
            self.right_q1_qpos,
            self.right_q1_dof,
            self.right_q4_qpos,
            self.right_q4_dof,
        )

        roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate = attitude(
            data.xquat[self.chassis],
            data.sensordata[self.chassis_gyro],
        )
        roll = float(roll)
        pitch = float(pitch)
        yaw = float(yaw)
        roll_rate = float(roll_rate)
        pitch_rate = float(pitch_rate)
        yaw_rate = float(yaw_rate)

        left_q1, left_q4, left_length, left_theta, left_dlength, left_dtheta = left
        right_q1, right_q4, right_length, right_theta, right_dlength, right_dtheta = right

        world_position = self._world_position(data)
        if self.last_world_position is None:
            self.reset(data)

        world_dx = world_position[0] - self.last_world_position[0]
        world_dy = world_position[1] - self.last_world_position[1]
        if self.last_yaw is None:
            heading_yaw = yaw
        else:
            yaw_step = math.atan2(
                math.sin(yaw - self.last_yaw),
                math.cos(yaw - self.last_yaw),
            )
            heading_yaw = self.last_yaw + yaw_step / 2.0

        forward_delta = (
            world_dx * math.cos(heading_yaw)
            + world_dy * math.sin(heading_yaw)
        )
        self.forward_position += forward_delta
        x = self.forward_position
        x_rate = forward_delta / self.model.opt.timestep
        self.last_world_position = world_position
        self.last_yaw = yaw

        return RobotState(
            left=LegState(
                q1=left_q1,
                q4=left_q4,
                length=left_length,
                theta=left_theta + pitch,
                length_rate=left_dlength,
                theta_rate=left_dtheta + pitch_rate,
            ),
            right=LegState(
                q1=right_q1,
                q4=right_q4,
                length=right_length,
                theta=right_theta + pitch,
                length_rate=right_dlength,
                theta_rate=right_dtheta + pitch_rate,
            ),
            roll=roll,
            roll_rate=roll_rate,
            pitch=pitch,
            pitch_rate=pitch_rate,
            yaw=yaw,
            yaw_rate=yaw_rate,
            x=x,
            x_rate=x_rate,
        )

    def write_command(self, data, command):
        if not isinstance(command, ActuatorCommand):
            raise TypeError("command must be an ActuatorCommand instance")

        data.ctrl[self.left_act_wheel] = command.left_wheel
        data.ctrl[self.left_act_q1] = command.left_q1
        data.ctrl[self.left_act_q4] = command.left_q4
        data.ctrl[self.right_act_wheel] = command.right_wheel
        data.ctrl[self.right_act_q1] = command.right_q1
        data.ctrl[self.right_act_q4] = command.right_q4
