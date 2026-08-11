"""
MuJoCo仿真入口。
加载 control_leg.xml 和 MATLAB LQR 参数，创建总控制器并以 1 ms 周期运行闭环仿真。
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
import mujoco.viewer

from simulation.controller import Controller


MODEL_PATH = ROOT / "model" / "control_leg.xml"
LQR_PATH = ROOT / "dsp" / "reference_lqr_build.mat"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    controller = Controller(model)
    controller.reset(data)

    last_print = 0.0
    last_time = data.time

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()

            if data.time < last_time:
                mujoco.mj_forward(model, data)
                controller.reset(data)
            last_time = data.time

            state = controller.update(data)
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.1:
                print(
                    f"t={data.time:6.3f}  "
                    f"L={state['L']:.4f}  "
                    f"theta={state['theta']:+.4f}  "
                    f"phi={state['phi']:+.4f}  "
                    f"x={state['x']:+.4f}  "
                    f"F={state['F']:+.2f}  "
                    f"T={state['T']:+.2f}  "
                    f"Tp={state['Tp']:+.2f}  "
                    f"tau=({state['tau1']:+.2f},{state['tau4']:+.2f})"
                )
                last_print = data.time

            sleep_time = model.opt.timestep - (time.perf_counter() - start)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()