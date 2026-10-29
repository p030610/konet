from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal, TypedDict

import torch


ProfilePhase = Literal[
    "data_wait", "h2d", "forward_loss", "backward", "optimizer", "validation", "checkpoint",
]


class ProfileTimingSummary(TypedDict):
    data_wait_seconds: float
    h2d_seconds: float
    forward_loss_seconds: float
    backward_seconds: float
    optimizer_seconds: float
    validation_seconds: float
    checkpoint_seconds: float
    training_phase_seconds: float
    unattributed_seconds: float
    measured_seconds: float
    wall_seconds: float


class ProfilePhaseRecorder:
    def __init__(self, device: torch.device, clock: Callable[[], float] = time.perf_counter) -> None:
        self._device = device
        self._clock = clock
        self._host_seconds = {phase: 0.0 for phase in (
            "data_wait", "h2d", "forward_loss", "backward", "optimizer", "validation", "checkpoint",
        )}
        self._cuda_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            phase: [] for phase in ("h2d", "forward_loss", "backward", "optimizer")
        }

    @property
    def pending_event_count(self) -> int:
        return sum(len(events) for events in self._cuda_events.values())

    @contextmanager
    def measure(self, phase: ProfilePhase) -> Iterator[None]:
        if self._device.type == "cuda" and phase in self._cuda_events:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            yield
            finished.record()
            self._cuda_events[phase].append((started, finished))
            return
        if self._device.type == "cuda" and phase == "validation":
            torch.cuda.synchronize(self._device)
        started_at = self._clock()
        yield
        if self._device.type == "cuda" and phase == "validation":
            torch.cuda.synchronize(self._device)
        self._host_seconds[phase] += max(0.0, self._clock() - started_at)

    def summary(self, measured_seconds: float) -> ProfileTimingSummary:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            for phase, events in self._cuda_events.items():
                self._host_seconds[phase] += sum(start.elapsed_time(finish) for start, finish in events) / 1000.0
                events.clear()
        training_seconds = sum(self._host_seconds[phase] for phase in (
            "data_wait", "h2d", "forward_loss", "backward", "optimizer",
        ))
        accounted_seconds = training_seconds + self._host_seconds["checkpoint"]
        return {
            "data_wait_seconds": self._host_seconds["data_wait"],
            "h2d_seconds": self._host_seconds["h2d"],
            "forward_loss_seconds": self._host_seconds["forward_loss"],
            "backward_seconds": self._host_seconds["backward"],
            "optimizer_seconds": self._host_seconds["optimizer"],
            "validation_seconds": self._host_seconds["validation"],
            "checkpoint_seconds": self._host_seconds["checkpoint"],
            "training_phase_seconds": training_seconds,
            "unattributed_seconds": max(0.0, measured_seconds - accounted_seconds),
            "measured_seconds": measured_seconds,
            "wall_seconds": measured_seconds,
        }


__all__ = ["ProfilePhaseRecorder", "ProfileTimingSummary"]
