"""
MuJoCo仿真入口。
加载 control_leg.xml，创建总控制器并以 1 ms 周期运行闭环仿真。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from control.controller import Controller
from control.targets import TargetStore
from simulation.keyboard_control import KeyboardTargetController
from simulation.mujoco_adapter import MujocoAdapter


MODEL_PATH = ROOT / "model" / "control_leg.xml"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    targets = TargetStore()
    controller = Controller(targets)
    adapter = MujocoAdapter(model)
    adapter.reset(data)
    controller.reset()

    last_print = 0.0
    last_time = data.time

    with (
        mujoco.viewer.launch_passive(model, data) as viewer,
        KeyboardTargetController(targets) as keyboard,
    ):
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = adapter.chassis
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        while viewer.is_running():
            start = time.perf_counter()

            if data.time < last_time:
                mujoco.mj_forward(model, data)
                adapter.reset(data)
                controller.reset()
                keyboard.reset()
            last_time = data.time

            target = keyboard.update(model.opt.timestep)
            state = adapter.read_state(data)
            command, telemetry = controller.update(
                state,
                model.opt.timestep,
            )
            adapter.write_command(data, command)
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.1:
                print(
                    f"t={data.time:6.3f}  "
                    f"L={telemetry.length:.4f}  "
                    f"theta={telemetry.theta:+.4f}  "
                    f"gamma={telemetry.roll:+.4f}  "
                    f"phi={telemetry.pitch:+.4f}  "
                    f"alpha={telemetry.yaw:+.4f}  "
                    f"x={telemetry.x:+.4f}  "
                    f"target_dx={target.x_rate:+.2f}  "
                    f"target_yaw={target.yaw:+.3f}  "
                    f"F={telemetry.common_force:+.2f}  "
                    f"T={telemetry.common_wheel_torque:+.2f}  "
                    f"Tp={telemetry.common_leg_torque:+.2f}  "
                    f"delta=({telemetry.delta_force:+.2f},"
                    f"{telemetry.delta_wheel_torque:+.2f},"
                    f"{telemetry.delta_leg_torque:+.2f})  "
                    f"left_tau=({telemetry.left_q1_torque:+.2f},"
                    f"{telemetry.left_q4_torque:+.2f})  "
                    f"right_tau=({telemetry.right_q1_torque:+.2f},"
                    f"{telemetry.right_q4_torque:+.2f})"
                )
                last_print = data.time

            sleep_time = model.opt.timestep - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
