"""Discrete-event virtual clock. FROZEN component.

Every action is charged here; wall-clock time never enters the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES = ("nav", "local", "upload", "edge", "wait")

# fixed action durations (Table 1)
MOVE_STEP_S = 0.5      # per 0.25 m
ROTATE_S = 0.3         # per rotate action


class DeadlineExceeded(Exception):
    pass


@dataclass
class VirtualClock:
    deadline: float
    now: float = 0.0
    spent: dict = field(default_factory=lambda: {c: 0.0 for c in CATEGORIES})

    def charge(self, dt: float, category: str, allow_overrun: bool = False) -> float:
        """Advance by dt. If the deadline is crossed, the clock stops AT the
        deadline and DeadlineExceeded is raised (unless allow_overrun)."""
        assert dt >= 0, dt
        assert category in CATEGORIES, category
        if not allow_overrun and self.now + dt > self.deadline + 1e-9:
            self.spent[category] += max(0.0, self.deadline - self.now)
            self.now = self.deadline
            raise DeadlineExceeded()
        self.now += dt
        self.spent[category] += dt
        return self.now

    def remaining(self) -> float:
        return max(0.0, self.deadline - self.now)
