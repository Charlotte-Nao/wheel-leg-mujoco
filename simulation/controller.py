"""
轮腿机器人总控制器。
读取 MuJoCo 状态，计算左右腿共模与差模控制量，并输出左右轮电机和髋关节电机力矩。
"""

import mujoco
import numpy as np

from dsp.attitude import update as attitude
from dsp.five_link import virtual_state
from dsp.leg_pd import LegPID
from dsp.lqr import LQR
from dsp.roll import update as roll
from dsp.theta_pd import update as theta_pd
from dsp.vmc import update as vmc
from dsp.yaw import update as yaw


class Controller:
    def __init__(self, model):
        self.model = model
        self.lqr = LQR()
        self.leg_pid = LegPID()

        self.left_q1_qpos, self.left_q1_dof = self._joint("left_q1")
        self.left_q4_qpos, self.left_q4_dof = self._joint("left_q4")
        self.right_q1_qpos, self.right_q1_dof = self._joint("right_q1")
        self.right_q4_qpos, self.right_q4_dof = self._joint("right_q4")

        self.chassis = self._id(mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.chassis_gyro = self._sensor("chassis_gyro")
        self.left_site_c = self._id(mujoco.mjtObj.mjOBJ_SITE, "left_C_wheel")
        self.right_site_c = self._id(mujoco.mjtObj.mjOBJ_SITE, "right_C_wheel")

        self.left_act_q1 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_q1_motor")
        self.left_act_q4 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_hip_q4_motor")
        self.left_act_wheel = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")

        self.right_act_q1 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_q1_motor")
        self.right_act_q4 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_hip_q4_motor")
        self.right_act_wheel = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")

        self.x_ref = None
        self.last_x = None

    def _id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return obj_id

    def _joint(self, name):
        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _sensor(self, name):
        sensor_id = self._id(mujoco.mjtObj.mjOBJ_SENSOR, name)
        address = int(self.model.sensor_adr[sensor_id])
        dimension = int(self.model.sensor_dim[sensor_id])
        return slice(address, address + dimension)

    def _leg_state(self, data, q1_qpos, q1_dof, q4_qpos, q4_dof):
        q1 = float(data.qpos[q1_qpos])
        q4 = float(data.qpos[q4_qpos])
        dq1 = float(data.qvel[q1_dof])
        dq4 = float(data.qvel[q4_dof])

        length, theta_body, length_rate, theta_body_rate = virtual_state(
            q1, q4, dq1, dq4
        )

        return q1, q4, length, theta_body, length_rate, theta_body_rate

    def _x_world(self, data):
        left_x = float(data.site_xpos[self.left_site_c, 0])
        right_x = float(data.site_xpos[self.right_site_c, 0])
        return (left_x + right_x) / 2.0

    def reset(self, data):
        x = self._x_world(data)
        self.x_ref = x
        self.last_x = x
        self.leg_pid.reset()

    def update(self, data):
        dt = self.model.opt.timestep

        left = self._leg_state(
            data,
            self.left_q1_qpos,
            self.left_q1_dof,
            self.left_q4_qpos,
            self.left_q4_dof,
        )
        right = self._leg_state(
            data,
            self.right_q1_qpos,
            self.right_q1_dof,
            self.right_q4_qpos,
            self.right_q4_dof,
        )

        left_q1, left_q4, left_length, left_theta_body, left_length_rate, left_theta_body_rate = left
        right_q1, right_q4, right_length, right_theta_body, right_length_rate, right_theta_body_rate = right

        gamma, phi, alpha, gamma_rate, phi_rate, alpha_rate = attitude(
            data.xquat[self.chassis],
            data.sensordata[self.chassis_gyro],
        )

        left_theta = left_theta_body + phi
        right_theta = right_theta_body + phi
        left_theta_rate = left_theta_body_rate + phi_rate
        right_theta_rate = right_theta_body_rate + phi_rate

        length = (left_length + right_length) / 2.0
        length_rate = (left_length_rate + right_length_rate) / 2.0
        theta = (left_theta + right_theta) / 2.0
        theta_rate = (left_theta_rate + right_theta_rate) / 2.0

        x_world = self._x_world(data)
        if self.x_ref is None or self.last_x is None:
            self.reset(data)

        x = x_world - self.x_ref
        x_rate = (x_world - self.last_x) / dt
        self.last_x = x_world

        state = np.array([
            theta,
            theta_rate,
            x,
            x_rate,
            phi,
            phi_rate,
        ])

        T_common, Tp_common = self.lqr.update(length, state)
        F_common = self.leg_pid.update(length, length_rate, dt)

        delta_T = yaw(alpha, alpha_rate)
        delta_Tp = theta_pd(
            left_theta - right_theta,
            left_theta_rate - right_theta_rate,
        )
        delta_length, delta_F = roll(
            gamma,
            left_length,
            right_length,
            left_length_rate,
            right_length_rate,
        )

        left_T = T_common + delta_T
        right_T = T_common - delta_T
        left_Tp = Tp_common + delta_Tp
        right_Tp = Tp_common - delta_Tp
        left_F = F_common + delta_F
        right_F = F_common - delta_F

        left_tau1, left_tau4 = vmc(left_F, left_Tp, left_q1, left_q4)
        right_tau1, right_tau4 = vmc(right_F, right_Tp, right_q1, right_q4)

        left_T = float(np.clip(left_T, -10.0, 10.0))
        right_T = float(np.clip(right_T, -10.0, 10.0))
        left_tau1 = float(np.clip(left_tau1, -40.0, 40.0))
        left_tau4 = float(np.clip(left_tau4, -40.0, 40.0))
        right_tau1 = float(np.clip(right_tau1, -40.0, 40.0))
        right_tau4 = float(np.clip(right_tau4, -40.0, 40.0))

        data.ctrl[self.left_act_wheel] = left_T
        data.ctrl[self.left_act_q1] = left_tau1
        data.ctrl[self.left_act_q4] = left_tau4

        data.ctrl[self.right_act_wheel] = right_T
        data.ctrl[self.right_act_q1] = right_tau1
        data.ctrl[self.right_act_q4] = right_tau4

        return {
            "L": length,
            "dL": length_rate,
            "theta": theta,
            "dtheta": theta_rate,
            "gamma": gamma,
            "dgamma": gamma_rate,
            "phi": phi,
            "dphi": phi_rate,
            "alpha": alpha,
            "dalpha": alpha_rate,
            "x": x,
            "dx": x_rate,
            "F": F_common,
            "T": T_common,
            "Tp": Tp_common,
            "delta_L": delta_length,
            "delta_F": delta_F,
            "delta_T": delta_T,
            "delta_Tp": delta_Tp,
            "left_L": left_length,
            "right_L": right_length,
            "left_theta": left_theta,
            "right_theta": right_theta,
            "left_F": left_F,
            "right_F": right_F,
            "left_T": left_T,
            "right_T": right_T,
            "left_Tp": left_Tp,
            "right_Tp": right_Tp,
            "left_tau1": left_tau1,
            "left_tau4": left_tau4,
            "right_tau1": right_tau1,
            "right_tau4": right_tau4,
        }
