"""Continuous multi-key target control for the MuJoCo viewer."""

import ctypes
import ctypes.util
from dataclasses import replace
import math
import time

from control.targets import TargetStore


MAX_X_RATE = 3.5
YAW_TARGET_RATE = 1.0
KEY_POLL_INTERVAL = 0.01
FOCUS_POLL_INTERVAL = 0.01
VIEWER_TITLE_PREFIX = "MuJoCo"
ANY_MODIFIER = 1 << 15
GRAB_MODE_ASYNC = 1
CURRENT_TIME = 0
KEY_PRESS = 2
KEY_RELEASE = 3


class XKeyEvent(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("root", ctypes.c_ulong),
        ("subwindow", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("x", ctypes.c_int),
        ("y", ctypes.c_int),
        ("x_root", ctypes.c_int),
        ("y_root", ctypes.c_int),
        ("state", ctypes.c_uint),
        ("keycode", ctypes.c_uint),
        ("same_screen", ctypes.c_int),
    ]


class XEvent(ctypes.Union):
    _fields_ = [
        ("type", ctypes.c_int),
        ("xkey", XKeyEvent),
        ("storage", ctypes.c_long * 24),
    ]


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class X11KeySource:
    """Poll key states while the MuJoCo X11 window has keyboard focus."""

    KEY_SYMBOLS = {
        "w": ord("w"),
        "a": ord("a"),
        "s": ord("s"),
        "d": ord("d"),
    }

    def __init__(self):
        library_name = ctypes.util.find_library("X11")
        if library_name is None:
            raise RuntimeError("Keyboard control requires the X11 library")

        self._x11 = ctypes.CDLL(library_name)
        self._configure_functions()
        self._display = self._x11.XOpenDisplay(None)
        if not self._display:
            raise RuntimeError("Keyboard control could not open the X11 display")

        self._keycodes = {
            name: int(self._x11.XKeysymToKeycode(self._display, symbol))
            for name, symbol in self.KEY_SYMBOLS.items()
        }
        self._pressed = frozenset()
        self._viewer_focused = False
        self._grabbed_window = None
        self._last_key_poll = -math.inf
        self._last_focus_poll = -math.inf

    def _configure_functions(self):
        self._x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self._x11.XOpenDisplay.restype = ctypes.c_void_p
        self._x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._x11.XCloseDisplay.restype = ctypes.c_int
        self._x11.XKeysymToKeycode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self._x11.XKeysymToKeycode.restype = ctypes.c_uint
        self._x11.XQueryKeymap.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char),
        ]
        self._x11.XQueryKeymap.restype = ctypes.c_int
        self._x11.XGrabKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._x11.XGrabKey.restype = ctypes.c_int
        self._x11.XUngrabKey.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_ulong,
        ]
        self._x11.XUngrabKey.restype = ctypes.c_int
        self._x11.XUngrabKeyboard.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
        ]
        self._x11.XUngrabKeyboard.restype = ctypes.c_int
        self._x11.XPending.argtypes = [ctypes.c_void_p]
        self._x11.XPending.restype = ctypes.c_int
        self._x11.XNextEvent.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(XEvent),
        ]
        self._x11.XNextEvent.restype = ctypes.c_int
        self._x11.XSync.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._x11.XSync.restype = ctypes.c_int
        self._x11.XGetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._x11.XGetInputFocus.restype = ctypes.c_int
        self._x11.XFetchName.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._x11.XFetchName.restype = ctypes.c_int
        self._x11.XFree.argtypes = [ctypes.c_void_p]
        self._x11.XFree.restype = ctypes.c_int

    def _viewer_focus_window(self):
        focus = ctypes.c_ulong()
        revert_to = ctypes.c_int()
        self._x11.XGetInputFocus(
            self._display,
            ctypes.byref(focus),
            ctypes.byref(revert_to),
        )
        if focus.value in (0, 1):
            return None

        window_name = ctypes.c_char_p()
        found = self._x11.XFetchName(
            self._display,
            focus.value,
            ctypes.byref(window_name),
        )
        if not found or not window_name.value:
            return None

        try:
            title = window_name.value.decode("utf-8", errors="replace")
        finally:
            self._x11.XFree(window_name)
        if not title.startswith(VIEWER_TITLE_PREFIX):
            return None
        return focus.value

    def _release_grabbed_keys(self):
        if self._grabbed_window is None:
            return

        for keycode in self._keycodes.values():
            self._x11.XUngrabKey(
                self._display,
                keycode,
                ANY_MODIFIER,
                self._grabbed_window,
            )
        self._x11.XUngrabKeyboard(self._display, CURRENT_TIME)
        self._x11.XSync(self._display, False)
        self._grabbed_window = None
        self._pressed = frozenset()

    def _query_pressed_keys(self):
        keymap = (ctypes.c_char * 32)()
        self._x11.XQueryKeymap(self._display, keymap)
        pressed = set()
        for name, keycode in self._keycodes.items():
            byte = keymap[keycode // 8][0]
            if byte & (1 << (keycode % 8)):
                pressed.add(name)
        return frozenset(pressed)

    def _set_grabbed_window(self, window):
        if window == self._grabbed_window:
            return

        self._release_grabbed_keys()
        if window is None:
            return

        pressed = self._query_pressed_keys()
        for keycode in self._keycodes.values():
            self._x11.XGrabKey(
                self._display,
                keycode,
                ANY_MODIFIER,
                window,
                False,
                GRAB_MODE_ASYNC,
                GRAB_MODE_ASYNC,
            )
        self._x11.XSync(self._display, False)
        self._grabbed_window = window
        self._pressed = pressed

    def _update_pressed_from_events(self):
        pressed = set(self._pressed)
        key_names = {
            keycode: name
            for name, keycode in self._keycodes.items()
        }
        event = XEvent()
        while self._x11.XPending(self._display):
            self._x11.XNextEvent(self._display, ctypes.byref(event))
            name = key_names.get(int(event.xkey.keycode))
            if name is None:
                continue
            if event.type == KEY_PRESS:
                pressed.add(name)
            elif event.type == KEY_RELEASE:
                pressed.discard(name)
        self._pressed = frozenset(pressed)

    def pressed_keys(self):
        now = time.monotonic()
        if now - self._last_key_poll < KEY_POLL_INTERVAL:
            return self._pressed
        self._last_key_poll = now

        if now - self._last_focus_poll >= FOCUS_POLL_INTERVAL:
            window = self._viewer_focus_window()
            self._set_grabbed_window(window)
            self._viewer_focused = window is not None
            self._last_focus_poll = now

        if not self._viewer_focused:
            self._pressed = frozenset()
            return self._pressed

        self._update_pressed_from_events()
        return self._pressed

    def close(self):
        if self._display:
            self._release_grabbed_keys()
            self._x11.XCloseDisplay(self._display)
            self._display = None


class KeyboardTargetController:
    """Convert held W/S/A/D keys into coherent control target trajectories."""

    def __init__(self, targets, key_source=None):
        if not isinstance(targets, TargetStore):
            raise TypeError("targets must be a TargetStore instance")

        self.targets = targets
        self.key_source = key_source if key_source is not None else X11KeySource()

    def update(self, dt):
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        keys = self.key_source.pressed_keys()
        forward_axis = int("w" in keys) - int("s" in keys)
        yaw_axis = int("a" in keys) - int("d" in keys)
        x_rate = forward_axis * MAX_X_RATE
        yaw_rate = yaw_axis * YAW_TARGET_RATE

        def advance(target):
            x = target.x + x_rate * dt
            yaw = wrap_angle(target.yaw + yaw_rate * dt)
            if (
                x == target.x
                and x_rate == target.x_rate
                and yaw == target.yaw
                and yaw_rate == target.yaw_rate
            ):
                return target
            return replace(
                target,
                x=x,
                x_rate=x_rate,
                yaw=yaw,
                yaw_rate=yaw_rate,
            )

        return self.targets.transform(advance)

    def reset(self):
        return self.targets.update(
            x=0.0,
            x_rate=0.0,
            yaw=0.0,
            yaw_rate=0.0,
        )

    def close(self):
        close = getattr(self.key_source, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
