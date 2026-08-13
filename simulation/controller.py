"""
轮腿机器人总控制器。
读取 MuJoCo 状态，依次执行左右五连杆解算、腿长控制、LQR、VMC，并输出左右轮电机和髋关节电机力矩。
"""

import mujoco
import numpy as np

from dsp.five_link import virtual_state
from dsp.leg_pd import update as leg_pd
from dsp.lqr import LQR
from dsp.vmc import update as vmc


class Controller:
    def __init__(self, model):
        self.model = model
        self.lqr = LQR()

        self.left_q1_qpos, self.left_q1_dof = self._joint("left_q1")
        self.left_q4_qpos, self.left_q4_dof = self._joint("left_q4")
        self.right_q1_qpos, self.right_q1_dof = self._joint("right_q1")
        self.right_q4_qpos, self.right_q4_dof = self._joint("right_q4")
        self.pitch_qpos, self.pitch_dof = self._joint("base_pitch")

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

    def _x_world(self, data):
        left_x = float(data.site_xpos[self.left_site_c, 0])
        right_x = float(data.site_xpos[self.right_site_c, 0])
        return (left_x + right_x) / 2.0

    def reset(self, data):
        x = self._x_world(data)
        self.x_ref = x
        self.last_x = x

    def update(self, data):
        dt = self.model.opt.timestep

        left_q1 = float(data.qpos[self.left_q1_qpos])
        left_q4 = float(data.qpos[self.left_q4_qpos])
        left_dq1 = float(data.qvel[self.left_q1_dof])
        left_dq4 = float(data.qvel[self.left_q4_dof])

        right_q1 = float(data.qpos[self.right_q1_qpos])
        right_q4 = float(data.qpos[self.right_q4_qpos])
        right_dq1 = float(data.qvel[self.right_q1_dof])
        right_dq4 = float(data.qvel[self.right_q4_dof])

        left_length, left_theta_body, left_length_rate, left_theta_body_rate = virtual_state(
            left_q1, left_q4, left_dq1, left_dq4
        )
        right_length, right_theta_body, right_length_rate, right_theta_body_rate = virtual_state(
            right_q1, right_q4, right_dq1, right_dq4
        )

        length = (left_length + right_length) / 2.0
        length_rate = (left_length_rate + right_length_rate) / 2.0

        x_world = self._x_world(data)
        if self.x_ref is None or self.last_x is None:
            self.reset(data)

        x = x_world - self.x_ref
        x_rate = (x_world - self.last_x) / dt
        self.last_x = x_world



        raw_qpos0 = float(self.model.qpos0[self.pitch_qpos])
        raw_phi_rate = float(data.qvel[self.pitch_dof])

        raw_phi = float(
            data.qpos[self.pitch_qpos]
            - raw_qpos0
        )

        left_theta = left_theta_body + raw_phi
        right_theta = right_theta_body + raw_phi
        left_theta_rate = left_theta_body_rate + raw_phi_rate
        right_theta_rate = right_theta_body_rate + raw_phi_rate

        theta = (left_theta + right_theta) / 2.0
        theta_rate = (left_theta_rate + right_theta_rate) / 2.0

        left_state = np.array([
            left_theta,
            left_theta_rate,
            x,
            x_rate,
            raw_phi,
            raw_phi_rate,
        ])
        right_state = np.array([
            right_theta,
            right_theta_rate,
            x,
            x_rate,
            raw_phi,
            raw_phi_rate,
        ])

        left_T, left_Tp = self.lqr.update(left_length, left_state)
        right_T, right_Tp = self.lqr.update(right_length, right_state)

        left_F = leg_pd(left_length, left_length_rate)
        right_F = leg_pd(right_length, right_length_rate)

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
            "phi": raw_phi,
            "dphi": raw_phi_rate,
            "x": x,
            "dx": x_rate,
            "F": (left_F + right_F) / 2.0,
            "T": (left_T + right_T) / 2.0,
            "Tp": (left_Tp + right_Tp) / 2.0,
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
