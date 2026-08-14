"""
MuJoCo车体六自由度状态编辑与显示。
"""

from queue import Empty, SimpleQueue

import glfw
import mujoco
import numpy as np

from dsp.attitude import update as attitude


POSITION_STEP = 0.02
ANGLE_STEP = np.deg2rad(2.0)
VELOCITY_STEP = 0.1
ANGLE_RATE_STEP = np.deg2rad(5.0)

AXIS_NAMES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
)


class StateEditor:
    def __init__(self, model):
        self.model = model
        self.command_queue = SimpleQueue()
        self.selected = 0
        self.paused = False

        joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "base_free")
        self.base_qpos = int(model.jnt_qposadr[joint_id])
        self.base_dof = int(model.jnt_dofadr[joint_id])

    def _id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return obj_id

    def key_callback(self, keycode):
        self.command_queue.put(keycode)

    def _state(self, data):
        position = data.qpos[self.base_qpos:self.base_qpos + 3].copy()
        quaternion = data.qpos[self.base_qpos + 3:self.base_qpos + 7]
        velocity = data.qvel[self.base_dof:self.base_dof + 3].copy()
        angular_velocity = data.qvel[self.base_dof + 3:self.base_dof + 6]

        gamma, phi, alpha, gamma_rate, phi_rate, alpha_rate = attitude(
            quaternion,
            angular_velocity,
        )

        pose = np.array([
            position[0],
            position[1],
            position[2],
            gamma,
            phi,
            alpha,
        ])
        rate = np.array([
            velocity[0],
            velocity[1],
            velocity[2],
            gamma_rate,
            phi_rate,
            alpha_rate,
        ])

        return pose, rate

    def _set_state(self, data, pose, rate):
        gamma, phi, alpha = pose[3:]
        gamma_rate, phi_rate, alpha_rate = rate[3:]

        cos_gamma = np.cos(gamma)
        sin_gamma = np.sin(gamma)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        angular_velocity = np.array([
            gamma_rate - alpha_rate * sin_phi,
            phi_rate * cos_gamma + alpha_rate * sin_gamma * cos_phi,
            -phi_rate * sin_gamma + alpha_rate * cos_gamma * cos_phi,
        ])

        cos_gamma = np.cos(gamma / 2.0)
        sin_gamma = np.sin(gamma / 2.0)
        cos_phi = np.cos(phi / 2.0)
        sin_phi = np.sin(phi / 2.0)
        cos_alpha = np.cos(alpha / 2.0)
        sin_alpha = np.sin(alpha / 2.0)

        quaternion = np.array([
            cos_gamma * cos_phi * cos_alpha + sin_gamma * sin_phi * sin_alpha,
            sin_gamma * cos_phi * cos_alpha - cos_gamma * sin_phi * sin_alpha,
            cos_gamma * sin_phi * cos_alpha + sin_gamma * cos_phi * sin_alpha,
            cos_gamma * cos_phi * sin_alpha - sin_gamma * sin_phi * cos_alpha,
        ])

        data.qpos[self.base_qpos:self.base_qpos + 3] = pose[:3]
        data.qpos[self.base_qpos + 3:self.base_qpos + 7] = quaternion
        data.qvel[self.base_dof:self.base_dof + 3] = rate[:3]
        data.qvel[self.base_dof + 3:self.base_dof + 6] = angular_velocity

    def _initial_pose(self):
        quaternion = self.model.qpos0[
            self.base_qpos + 3:self.base_qpos + 7
        ]
        gamma, phi, alpha, _, _, _ = attitude(quaternion, np.zeros(3))

        return np.array([
            self.model.qpos0[self.base_qpos],
            self.model.qpos0[self.base_qpos + 1],
            self.model.qpos0[self.base_qpos + 2],
            gamma,
            phi,
            alpha,
        ])

    def update(self, data):
        state_changed = False
        reset = False

        while True:
            try:
                keycode = self.command_queue.get_nowait()
            except Empty:
                break

            if glfw.KEY_1 <= keycode <= glfw.KEY_6:
                self.selected = keycode - glfw.KEY_1
                continue

            if keycode == glfw.KEY_P:
                self.paused = not self.paused
                continue

            pose, rate = self._state(data)

            if keycode in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                step = POSITION_STEP if self.selected < 3 else ANGLE_STEP
                pose[self.selected] += step
            elif keycode in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                step = POSITION_STEP if self.selected < 3 else ANGLE_STEP
                pose[self.selected] -= step
            elif keycode == glfw.KEY_RIGHT_BRACKET:
                step = VELOCITY_STEP if self.selected < 3 else ANGLE_RATE_STEP
                rate[self.selected] += step
            elif keycode == glfw.KEY_LEFT_BRACKET:
                step = VELOCITY_STEP if self.selected < 3 else ANGLE_RATE_STEP
                rate[self.selected] -= step
            elif keycode in (glfw.KEY_0, glfw.KEY_KP_0):
                pose[self.selected] = self._initial_pose()[self.selected]
                rate[self.selected] = 0.0
            elif keycode == glfw.KEY_BACKSPACE:
                pose = self._initial_pose()
                rate = np.zeros(6)
                reset = True
            else:
                continue

            pose[3] = np.arctan2(np.sin(pose[3]), np.cos(pose[3]))
            pose[4] = np.clip(
                pose[4],
                -np.deg2rad(89.0),
                np.deg2rad(89.0),
            )
            pose[5] = np.arctan2(np.sin(pose[5]), np.cos(pose[5]))

            self._set_state(data, pose, rate)
            state_changed = True

        if state_changed:
            mujoco.mj_forward(self.model, data)

        return state_changed, reset

    def texts(self, data):
        pose, rate = self._state(data)

        pose_values = (
            f"{pose[0]:+8.3f} m",
            f"{pose[1]:+8.3f} m",
            f"{pose[2]:+8.3f} m",
            f"{np.rad2deg(pose[3]):+8.2f} deg",
            f"{np.rad2deg(pose[4]):+8.2f} deg",
            f"{np.rad2deg(pose[5]):+8.2f} deg",
        )
        rate_values = (
            f"{rate[0]:+8.3f} m/s",
            f"{rate[1]:+8.3f} m/s",
            f"{rate[2]:+8.3f} m/s",
            f"{np.rad2deg(rate[3]):+8.2f} deg/s",
            f"{np.rad2deg(rate[4]):+8.2f} deg/s",
            f"{np.rad2deg(rate[5]):+8.2f} deg/s",
        )

        labels = []
        values = []
        for index, name in enumerate(AXIS_NAMES):
            marker = ">" if index == self.selected else " "
            labels.append(f"{marker} {index + 1}  {name}")
            values.append(f"{pose_values[index]}   {rate_values[index]}")

        run_state = "PAUSED" if self.paused else "RUNNING"
        title = f"STATE EDITOR  [{run_state}]\nposition / angle"
        value_title = "\nvelocity / angle rate"
        controls = (
            "1-6: select axis\n"
            "-/=: position or angle -/+\n"
            "[ / ]: velocity or angle rate -/+\n"
            "0: reset selected axis\n"
            "Backspace: reset base   P: pause"
        )

        return [
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                title + "\n" + "\n".join(labels),
                value_title + "\n" + "\n".join(values),
            ),
            (
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                controls,
                "",
            ),
        ]
