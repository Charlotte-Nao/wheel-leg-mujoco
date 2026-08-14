import math
from dataclasses import astuple
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np

from control.controller import Controller, angle_error
from control.targets import ControlTargets, TargetStore
from control.types import LegState, RobotState
from dsp.leg_pd import FF, KD, KI, KP, LegPID
from simulation.keyboard_control import (
    ANY_MODIFIER,
    CURRENT_TIME,
    GRAB_MODE_ASYNC,
    KEY_PRESS,
    KEY_RELEASE,
    MAX_X_RATE,
    YAW_TARGET_RATE,
    KeyboardTargetController,
    X11KeySource,
)
from simulation.mujoco_adapter import MujocoAdapter


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "control_leg.xml"


class ControlTargetsTest(unittest.TestCase):
    def test_defaults_match_existing_zero_regulators(self):
        target = ControlTargets()

        self.assertEqual(target.lqr_state, (0.0,) * 6)
        self.assertEqual(target.yaw, 0.0)
        self.assertEqual(target.roll, 0.0)
        self.assertEqual(target.leg_length, 0.2)
        self.assertEqual(target.theta_difference, 0.0)

    def test_store_replaces_an_immutable_snapshot(self):
        store = TargetStore()
        old_snapshot = store.snapshot()

        new_snapshot = store.update(x=1.25, x_rate=0.4, yaw=-0.2)

        self.assertEqual(old_snapshot.x, 0.0)
        self.assertEqual(new_snapshot.x, 1.25)
        self.assertEqual(new_snapshot.x_rate, 0.4)
        self.assertEqual(new_snapshot.yaw, -0.2)
        self.assertIs(new_snapshot, store.snapshot())

    def test_rejects_invalid_targets(self):
        with self.assertRaises(ValueError):
            ControlTargets(x=math.nan)
        with self.assertRaises(ValueError):
            ControlTargets(leg_length=0.01)
        with self.assertRaises(TypeError):
            TargetStore().set(object())


class LegPIDTest(unittest.TestCase):
    def test_explicit_length_and_rate_references(self):
        controller = LegPID()
        dt = 0.001
        length = 0.2
        length_ref = 0.21
        length_rate = -0.02
        length_rate_ref = 0.03

        left, right, delta = controller.update(
            roll_error=0.0,
            left_length=length,
            right_length=length,
            left_length_rate=length_rate,
            right_length_rate=length_rate,
            length_ref=length_ref,
            length_rate_ref=length_rate_ref,
            dt=dt,
        )

        expected = (
            FF
            + KP * (length_ref - length)
            + KD * (length_rate_ref - length_rate)
            + KI * (length_ref - length) * dt
        )
        self.assertAlmostEqual(left, expected)
        self.assertAlmostEqual(right, expected)
        self.assertEqual(delta, 0.0)

    def test_roll_correction_is_added_to_current_length_difference(self):
        controller = LegPID()
        dt = 0.001
        left_length = 0.16
        right_length = 0.24
        length_ref = 0.2
        roll_error = 0.1

        left, right, delta = controller.update(
            roll_error=roll_error,
            left_length=left_length,
            right_length=right_length,
            left_length_rate=0.0,
            right_length_rate=0.0,
            length_ref=length_ref,
            dt=dt,
        )

        expected_delta = 0.5 * math.sin(roll_error)
        current_difference = right_length - left_length
        target_difference = current_difference + expected_delta
        left_ref = length_ref - target_difference / 2.0
        right_ref = length_ref + target_difference / 2.0
        expected_left = FF + (KP + KI * dt) * (left_ref - left_length)
        expected_right = FF + (KP + KI * dt) * (right_ref - right_length)

        self.assertAlmostEqual(delta, expected_delta)
        self.assertAlmostEqual(left, expected_left)
        self.assertAlmostEqual(right, expected_right)


class ControllerErrorTest(unittest.TestCase):
    def test_targets_are_converted_to_expected_feedback_errors(self):
        targets = TargetStore(ControlTargets(
            theta=0.1,
            theta_rate=0.2,
            x=1.0,
            x_rate=0.4,
            pitch=-0.1,
            pitch_rate=-0.2,
            yaw=math.radians(-179.0),
            yaw_rate=0.3,
            roll=0.05,
            leg_length=0.23,
            leg_length_rate=0.01,
            theta_difference=0.12,
            theta_difference_rate=0.04,
        ))
        controller = Controller(targets)
        controller.lqr.update = Mock(return_value=(1.0, 2.0))
        controller.leg_pid.update = Mock(return_value=(100.0, 100.0, 0.0))

        state = RobotState(
            left=LegState(2.0, 1.0, 0.22, 0.3, 0.02, 0.08),
            right=LegState(2.0, 1.0, 0.24, 0.1, -0.02, 0.01),
            roll=0.08,
            roll_rate=0.0,
            pitch=-0.04,
            pitch_rate=-0.05,
            yaw=math.radians(179.0),
            yaw_rate=0.5,
            x=1.4,
            x_rate=0.7,
        )

        with (
            patch("control.controller.yaw", return_value=0.0) as yaw_mock,
            patch("control.controller.roll", return_value=0.0) as roll_mock,
            patch("control.controller.theta_pd", return_value=0.0) as theta_mock,
            patch("control.controller.vmc", side_effect=[(3.0, 4.0), (5.0, 6.0)]) as vmc_mock,
        ):
            command, telemetry = controller.update(state, 0.001)

        gain_length, lqr_error = controller.lqr.update.call_args.args
        self.assertAlmostEqual(gain_length, 0.23)
        np.testing.assert_allclose(
            lqr_error,
            [0.1, -0.155, 0.4, 0.3, 0.06, 0.15],
        )
        controller.leg_pid.update.assert_called_once_with(
            roll_error=0.03,
            left_length=0.22,
            right_length=0.24,
            left_length_rate=0.02,
            right_length_rate=-0.02,
            length_ref=0.23,
            length_rate_ref=0.01,
            dt=0.001,
        )
        self.assertAlmostEqual(yaw_mock.call_args.args[0], math.radians(-2.0))
        self.assertAlmostEqual(yaw_mock.call_args.args[1], 0.2)
        roll_mock.assert_called_once_with(0.03)
        self.assertAlmostEqual(theta_mock.call_args.args[0], 0.08)
        self.assertAlmostEqual(theta_mock.call_args.args[1], 0.03)
        self.assertEqual(vmc_mock.call_count, 2)
        self.assertEqual(command.left_q1, 3.0)
        self.assertEqual(command.right_q4, 6.0)
        self.assertAlmostEqual(telemetry.roll_error, 0.03)

    def test_angle_error_uses_shortest_path(self):
        error = angle_error(
            math.radians(179.0),
            math.radians(-179.0),
        )
        self.assertAlmostEqual(error, math.radians(-2.0))


class FakeKeySource:
    def __init__(self, *keys):
        self.keys = set(keys)
        self.closed = False

    def pressed_keys(self):
        return frozenset(self.keys)

    def close(self):
        self.closed = True


class X11KeyGrabTest(unittest.TestCase):
    def setUp(self):
        self.source = X11KeySource.__new__(X11KeySource)
        self.source._x11 = Mock()
        self.source._display = object()
        self.source._keycodes = {
            "w": 25,
            "a": 38,
            "s": 39,
            "d": 40,
        }
        self.source._grabbed_window = None
        self.source._pressed = frozenset()

    def test_grabs_control_keys_without_forwarding_to_viewer(self):
        self.source._set_grabbed_window(1234)

        self.assertEqual(self.source._x11.XGrabKey.call_count, 4)
        self.source._x11.XGrabKey.assert_any_call(
            self.source._display,
            25,
            ANY_MODIFIER,
            1234,
            False,
            GRAB_MODE_ASYNC,
            GRAB_MODE_ASYNC,
        )
        self.assertEqual(self.source._grabbed_window, 1234)

    def test_focus_change_releases_previous_window_grabs(self):
        self.source._set_grabbed_window(1234)
        self.source._set_grabbed_window(5678)

        self.assertEqual(self.source._x11.XUngrabKey.call_count, 4)
        self.source._x11.XUngrabKeyboard.assert_called_once_with(
            self.source._display,
            CURRENT_TIME,
        )
        self.assertEqual(self.source._grabbed_window, 5678)

    def test_close_releases_grabs_before_closing_display(self):
        self.source._set_grabbed_window(1234)
        display = self.source._display
        self.source.close()

        self.assertEqual(self.source._x11.XUngrabKey.call_count, 4)
        self.source._x11.XCloseDisplay.assert_called_once_with(
            display,
        )
        self.assertIsNone(self.source._display)

    def test_grabbed_events_drive_multi_key_and_release_state(self):
        events = [
            (KEY_PRESS, 25),
            (KEY_PRESS, 38),
        ]
        self.source._x11.XPending.side_effect = [1, 1, 0]

        def next_event(_display, event_pointer):
            event_type, keycode = events.pop(0)
            event = event_pointer._obj
            event.type = event_type
            event.xkey.keycode = keycode

        self.source._x11.XNextEvent.side_effect = next_event
        self.source._update_pressed_from_events()

        self.assertEqual(self.source._pressed, frozenset(("w", "a")))

        self.source._x11.XPending.side_effect = [1, 0]
        self.source._x11.XNextEvent.side_effect = lambda _display, pointer: (
            setattr(pointer._obj, "type", KEY_RELEASE),
            setattr(pointer._obj.xkey, "keycode", 25),
        )
        self.source._update_pressed_from_events()

        self.assertEqual(self.source._pressed, frozenset(("a",)))


class KeyboardTargetControllerTest(unittest.TestCase):
    def setUp(self):
        self.targets = TargetStore()
        self.keys = FakeKeySource()
        self.keyboard = KeyboardTargetController(self.targets, self.keys)

    def test_combined_keys_and_long_press_update_both_trajectories(self):
        self.keys.keys = {"w", "a"}

        first = self.keyboard.update(0.25)
        second = self.keyboard.update(0.25)

        self.assertEqual(first.x_rate, MAX_X_RATE)
        self.assertEqual(first.yaw_rate, YAW_TARGET_RATE)
        self.assertAlmostEqual(first.x, 0.25 * MAX_X_RATE)
        self.assertAlmostEqual(first.yaw, 0.25 * YAW_TARGET_RATE)
        self.assertAlmostEqual(second.x, 0.5 * MAX_X_RATE)
        self.assertAlmostEqual(second.yaw, 0.5 * YAW_TARGET_RATE)
        self.assertEqual(MAX_X_RATE, 3.5)

    def test_release_stops_rates_and_holds_position_targets(self):
        self.keys.keys = {"w", "a"}
        moving = self.keyboard.update(0.2)
        self.keys.keys.clear()

        stopped = self.keyboard.update(0.1)

        self.assertEqual(stopped.x_rate, 0.0)
        self.assertEqual(stopped.yaw_rate, 0.0)
        self.assertEqual(stopped.x, moving.x)
        self.assertEqual(stopped.yaw, moving.yaw)

    def test_opposite_keys_cancel_and_reverse_keys_are_negative(self):
        self.keys.keys = {"w", "s", "a", "d"}
        cancelled = self.keyboard.update(0.1)
        self.assertEqual(cancelled.x_rate, 0.0)
        self.assertEqual(cancelled.yaw_rate, 0.0)

        self.keys.keys = {"s", "d"}
        reverse = self.keyboard.update(0.2)
        self.assertEqual(reverse.x_rate, -MAX_X_RATE)
        self.assertEqual(reverse.yaw_rate, -YAW_TARGET_RATE)
        self.assertAlmostEqual(reverse.x, -0.2 * MAX_X_RATE)
        self.assertAlmostEqual(reverse.yaw, -0.2 * YAW_TARGET_RATE)

    def test_yaw_target_wraps_at_pi(self):
        self.targets.update(yaw=math.pi - 0.05)
        self.keys.keys = {"a"}

        target = self.keyboard.update(0.1)

        self.assertAlmostEqual(target.yaw, -math.pi + 0.05)

    def test_reset_and_close(self):
        self.targets.update(x=2.0, x_rate=0.5, yaw=1.0, yaw_rate=0.2)

        target = self.keyboard.reset()
        self.keyboard.close()

        self.assertEqual(target.x, 0.0)
        self.assertEqual(target.x_rate, 0.0)
        self.assertEqual(target.yaw, 0.0)
        self.assertEqual(target.yaw_rate, 0.0)
        self.assertTrue(self.keys.closed)


class TerrainModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    @classmethod
    def _geom_id(cls, name):
        geom_id = mujoco.mj_name2id(
            cls.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            name,
        )
        if geom_id < 0:
            raise AssertionError(f"Missing terrain geom: {name}")
        return geom_id

    @classmethod
    def _top_endpoints(cls, geom_id):
        rotation = cls.data.geom_xmat[geom_id].reshape(3, 3)
        half_length, _, half_thickness = cls.model.geom_size[geom_id]
        centre = cls.model.geom_pos[geom_id]
        low = centre + rotation @ np.array([
            -half_length,
            0.0,
            half_thickness,
        ])
        high = centre + rotation @ np.array([
            half_length,
            0.0,
            half_thickness,
        ])
        return low, high

    def test_single_wheel_ramp_only_covers_left_track(self):
        geom_id = self._geom_id("left_wheel_ramp")
        centre_y = self.model.geom_pos[geom_id, 1]
        half_width = self.model.geom_size[geom_id, 1]
        low, high = self._top_endpoints(geom_id)

        self.assertLessEqual(abs(0.25 - centre_y), half_width)
        self.assertGreater(abs(-0.25 - centre_y), half_width)
        self.assertLess(low[2], 0.025)
        self.assertGreater(high[2], 0.27)
        self.assertEqual(self.model.geom_contype[geom_id], 1)
        self.assertEqual(self.model.geom_conaffinity[geom_id], 0)

    def test_launch_ramp_covers_both_tracks_and_has_drop(self):
        geom_id = self._geom_id("launch_ramp")
        centre_y = self.model.geom_pos[geom_id, 1]
        half_width = self.model.geom_size[geom_id, 1]
        low, high = self._top_endpoints(geom_id)

        self.assertLessEqual(abs(0.25 - centre_y), half_width)
        self.assertLessEqual(abs(-0.25 - centre_y), half_width)
        self.assertLess(low[2], 0.025)
        self.assertGreater(high[2], 0.43)
        self.assertEqual(self.model.geom_contype[geom_id], 1)
        self.assertEqual(self.model.geom_conaffinity[geom_id], 0)


class TerrainTraversalTest(unittest.TestCase):
    def test_maximum_speed_course_hits_expected_wheels(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        targets = TargetStore()
        keyboard = KeyboardTargetController(
            targets,
            FakeKeySource("w"),
        )
        adapter = MujocoAdapter(model)
        controller = Controller(targets)
        adapter.reset(data)
        controller.reset()

        geom_ids = {
            name: mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                name,
            )
            for name in (
                "left_wheel_geom",
                "right_wheel_geom",
                "left_wheel_ramp",
                "launch_ramp",
            )
        }
        contacts = {
            "left_side": False,
            "right_side": False,
            "left_launch": False,
            "right_launch": False,
        }
        expected_pairs = {
            "left_side": frozenset((
                geom_ids["left_wheel_geom"],
                geom_ids["left_wheel_ramp"],
            )),
            "right_side": frozenset((
                geom_ids["right_wheel_geom"],
                geom_ids["left_wheel_ramp"],
            )),
            "left_launch": frozenset((
                geom_ids["left_wheel_geom"],
                geom_ids["launch_ramp"],
            )),
            "right_launch": frozenset((
                geom_ids["right_wheel_geom"],
                geom_ids["launch_ramp"],
            )),
        }
        maximum_roll = 0.0
        maximum_single_ramp_roll = 0.0
        maximum_height = float(data.qpos[2])

        for _ in range(3500):
            target = keyboard.update(model.opt.timestep)
            state = adapter.read_state(data)
            command, telemetry = controller.update(
                state,
                model.opt.timestep,
            )
            adapter.write_command(data, command)
            mujoco.mj_step(model, data)

            maximum_roll = max(maximum_roll, abs(telemetry.roll))
            maximum_height = max(maximum_height, float(data.qpos[2]))
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                pair = frozenset((int(contact.geom1), int(contact.geom2)))
                for name, expected_pair in expected_pairs.items():
                    if pair == expected_pair:
                        contacts[name] = True
                        if name == "left_side":
                            maximum_single_ramp_roll = max(
                                maximum_single_ramp_roll,
                                abs(telemetry.roll),
                            )

            self.assertTrue(np.isfinite((
                target.x,
                telemetry.x,
                telemetry.roll,
                telemetry.pitch,
                *data.ctrl,
            )).all())

        self.assertTrue(contacts["left_side"])
        self.assertFalse(contacts["right_side"])
        self.assertTrue(contacts["left_launch"])
        self.assertTrue(contacts["right_launch"])
        self.assertLess(maximum_single_ramp_roll, 0.15)
        self.assertGreater(maximum_roll, 0.1)
        self.assertGreater(maximum_height, 0.65)


class MujocoIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.targets = TargetStore()
        self.controller = Controller(self.targets)
        self.adapter = MujocoAdapter(self.model)
        self.adapter.reset(self.data)
        self.controller.reset()

    def _step(self):
        state = self.adapter.read_state(self.data)
        command, telemetry = self.controller.update(
            state,
            self.model.opt.timestep,
        )
        self.adapter.write_command(self.data, command)
        mujoco.mj_step(self.model, self.data)
        return command, telemetry

    def test_forward_odometry_remains_positive_beyond_ninety_degree_yaw(self):
        yaw = math.radians(120.0)
        half_yaw = yaw / 2.0
        self.data.qpos[3:7] = [
            math.cos(half_yaw),
            0.0,
            0.0,
            math.sin(half_yaw),
        ]
        mujoco.mj_forward(self.model, self.data)
        self.adapter.reset(self.data)
        initial = self.adapter.read_state(self.data)

        distance = 0.01
        self.data.qpos[0] += distance * math.cos(yaw)
        self.data.qpos[1] += distance * math.sin(yaw)
        mujoco.mj_forward(self.model, self.data)
        forward = self.adapter.read_state(self.data)

        self.assertAlmostEqual(initial.x, 0.0)
        self.assertLess(self.data.qpos[0], 0.0)
        self.assertAlmostEqual(forward.x, distance, places=12)
        self.assertAlmostEqual(
            forward.x_rate,
            distance / self.model.opt.timestep,
            places=10,
        )

        lateral_distance = 0.01
        self.data.qpos[0] -= lateral_distance * math.sin(yaw)
        self.data.qpos[1] += lateral_distance * math.cos(yaw)
        mujoco.mj_forward(self.model, self.data)
        lateral = self.adapter.read_state(self.data)

        self.assertAlmostEqual(lateral.x, distance, places=12)
        self.assertAlmostEqual(lateral.x_rate, 0.0, places=10)

    def test_default_targets_match_previous_1000_step_result(self):
        for _ in range(1000):
            command, telemetry = self._step()

        self.assertAlmostEqual(telemetry.length, 0.20553044211881683, places=12)
        self.assertAlmostEqual(telemetry.theta, 0.00021240210296012251, places=12)
        self.assertAlmostEqual(telemetry.x, -0.0007506588672550607, places=12)
        self.assertAlmostEqual(telemetry.common_force, 98.1950112684564, places=10)
        self.assertAlmostEqual(command.left_wheel, 8.589836149383902e-05, places=12)
        self.assertAlmostEqual(command.left_q1, -13.805274741505174, places=10)
        self.assertAlmostEqual(command.left_q4, 13.805595422107572, places=10)

    def test_runtime_target_update_reaches_controller(self):
        state = self.adapter.read_state(self.data)
        self.targets.update(
            x=0.05,
            x_rate=0.1,
            yaw=0.02,
            leg_length=0.21,
        )

        command, telemetry = self.controller.update(
            state,
            self.model.opt.timestep,
        )

        self.assertAlmostEqual(telemetry.lqr_error[2], -0.05)
        self.assertAlmostEqual(telemetry.lqr_error[3], -0.1)
        self.assertAlmostEqual(telemetry.yaw_error, -0.02)
        self.assertTrue(np.isfinite(astuple(command)).all())


if __name__ == "__main__":
    unittest.main()
