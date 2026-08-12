"""
轮腿机器人总控制器。
读取 MuJoCo 状态，依次执行五连杆解算、腿长控制、LQR、VMC，并输出轮电机和两个髋关节电机力矩。
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

        self.q1_qpos, self.q1_dof = self._joint("q1")
        self.q4_qpos, self.q4_dof = self._joint("q4")
        self.pitch_qpos, self.pitch_dof = self._joint("base_pitch")

        self.site_c = self._id(mujoco.mjtObj.mjOBJ_SITE, "C_left")
        self.act_q1 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_q1_motor")
        self.act_q4 = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_q4_motor")
        self.act_wheel = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_motor")

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

    def reset(self, data):
        x = float(data.site_xpos[self.site_c, 0])
        self.x_ref = x
        self.last_x = x

    def update(self, data):
        dt = self.model.opt.timestep

        q1 = float(data.qpos[self.q1_qpos])
        q4 = float(data.qpos[self.q4_qpos])
        dq1 = float(data.qvel[self.q1_dof])
        dq4 = float(data.qvel[self.q4_dof])

        length, theta, length_rate, theta_rate = virtual_state(q1, q4, dq1, dq4)

        x_world = float(data.site_xpos[self.site_c, 0])
        if self.x_ref is None or self.last_x is None:
            self.reset(data)

        x = x_world - self.x_ref
        x_rate = (x_world - self.last_x) / dt
        self.last_x = x_world



        raw_qpos0 = float(self.model.qpos0[self.pitch_qpos])
        raw_theta_rate = float(data.qvel[self.pitch_dof])

        raw_theta = float(
            data.qpos[self.pitch_qpos]
            - raw_qpos0
        )

        PHI_SIGN = - 1.0  # 暂时先这样，下面通过实验确认

        theta = PHI_SIGN * raw_theta
        theta_rate = PHI_SIGN * raw_theta_rate

        state = np.array([theta, theta_rate, x, x_rate, theta, theta_rate])

        T, Tp = self.lqr.update(length, state)

        F = leg_pd(length, length_rate)
        tau1, tau4 = vmc(F, Tp, q1, q4)

        T = float(np.clip(T, -10.0, 10.0))
        tau1 = float(np.clip(tau1, -40.0, 40.0))
        tau4 = float(np.clip(tau4, -40.0, 40.0))

        data.ctrl[self.act_wheel] = T
        data.ctrl[self.act_q1] = tau1
        data.ctrl[self.act_q4] = tau4

        return {
            "L": length,
            "dL": length_rate,
            "theta": theta,
            "dtheta": theta_rate,
            "x": x,
            "dx": x_rate,
            "F": F,
            "T": T,
            "Tp": Tp,
            "tau1": tau1,
            "tau4": tau4,
        }