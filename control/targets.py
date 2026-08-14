"""Runtime control targets and their synchronized shared store."""

from dataclasses import dataclass, replace
import math
from threading import Lock


LEG_LENGTH_MIN = 0.04
LEG_LENGTH_MAX = 0.46


@dataclass(frozen=True, slots=True)
class ControlTargets:
    """Physical references for every feedback loop.

    Angles are in radians, angular rates are in radians per second, lengths
    are in metres, and linear rates are in metres per second. ``x`` is signed
    distance accumulated along the platform's forward heading since reset.
    """

    theta: float = 0.0
    theta_rate: float = 0.0
    x: float = 0.0
    x_rate: float = 0.0
    pitch: float = 0.0
    pitch_rate: float = 0.0
    yaw: float = 0.0
    yaw_rate: float = 0.0
    roll: float = 0.0
    leg_length: float = 0.2
    leg_length_rate: float = 0.0
    theta_difference: float = 0.0
    theta_difference_rate: float = 0.0

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"Control target {name} must be finite")

        if not LEG_LENGTH_MIN <= self.leg_length <= LEG_LENGTH_MAX:
            raise ValueError(
                "leg_length must be within "
                f"[{LEG_LENGTH_MIN}, {LEG_LENGTH_MAX}] metres"
            )

    @property
    def lqr_state(self):
        return (
            self.theta,
            self.theta_rate,
            self.x,
            self.x_rate,
            self.pitch,
            self.pitch_rate,
        )


class TargetStore:
    """Owns the current immutable target snapshot.

    Writers replace a complete validated snapshot while the controller reads
    one coherent snapshot per cycle.
    """

    def __init__(self, targets=None):
        self._lock = Lock()
        self._targets = targets if targets is not None else ControlTargets()
        if not isinstance(self._targets, ControlTargets):
            raise TypeError("targets must be a ControlTargets instance")

    def snapshot(self):
        with self._lock:
            return self._targets

    def set(self, targets):
        if not isinstance(targets, ControlTargets):
            raise TypeError("targets must be a ControlTargets instance")

        with self._lock:
            self._targets = targets
            return self._targets

    def update(self, **changes):
        with self._lock:
            self._targets = replace(self._targets, **changes)
            return self._targets

    def transform(self, transform):
        """Atomically replace the target using its current snapshot."""

        if not callable(transform):
            raise TypeError("transform must be callable")

        with self._lock:
            targets = transform(self._targets)
            if not isinstance(targets, ControlTargets):
                raise TypeError("transform must return a ControlTargets instance")
            self._targets = targets
            return self._targets
