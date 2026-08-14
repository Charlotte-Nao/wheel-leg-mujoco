"""Platform-independent wheel-legged robot control package."""

from control.controller import Controller
from control.targets import ControlTargets, TargetStore

__all__ = ["Controller", "ControlTargets", "TargetStore"]
