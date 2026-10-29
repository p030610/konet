from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Literal, Protocol

import torch
from torch import nn
from torch.nn import functional as functional

from .errors import TrainingError
from .timing import ProfilePhaseRecorder


class SyncModel(Protocol):
    def __call__(self, visual: torch.Tensor, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...

    def parameters(self, recurse: bool = True) -> list[nn.Parameter]: ...

    def train(self, mode: bool = True) -> nn.Module: ...


@dataclass(frozen=True, slots=True)
class TrainConfig:
    microbatch_size: int
    effective_batch_size: int
    max_grad_norm: float
    amp_fp16: bool = False

    def __post_init__(self) -> None:
        if self.microbatch_size < 1 or self.effective_batch_size < 1:
            raise TrainingError("batch sizes must be positive")
        if self.max_grad_norm <= 0:
            raise TrainingError("max_grad_norm must be positive")


@dataclass(frozen=True, slots=True)
class TrainState:
    epoch: int = 0
    next_example_cursor: int = 0
    global_update: int = 0
    examples_seen: int = 0
    best_validation_loss: float | None = None
    best_epoch: int | None = None
    best_primary_signed80: float | None = None
    best_balanced_bce: float | None = None


@dataclass(frozen=True, slots=True)
class UpdateMetric:
    update: int
    loss: float
    pair_count: int
    grad_norm: float


@dataclass(frozen=True, slots=True)
class TrainOutcome:
    state: TrainState
    metrics: tuple[UpdateMetric, ...]


def create_grad_scaler(device_type: Literal["cpu", "cuda"], *, enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(device_type, init_scale=1.0, enabled=enabled)


def cosine_bce_loss(
    model: SyncModel,
    visual: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
    amp: bool,
    require_gradient: bool = True,
) -> torch.Tensor:
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else nullcontext()
    with context:
        vision_embeddings, audio_embeddings = model(visual, audio)
    similarities = functional.cosine_similarity(vision_embeddings.float(), audio_embeddings.float())
    if bool(((similarities < -1e-6) | (similarities > 1.0 + 1e-6)).any().item()):
        raise TrainingError("cosine similarity is outside BCE probability domain")
    probabilities = similarities.clamp(0.0, 1.0)
    if not bool(torch.isfinite(probabilities).all().item()):
        raise TrainingError("non-finite cosine probability")
    loss = functional.binary_cross_entropy(probabilities, labels.float(), reduction="sum")
    if not torch.isfinite(loss):
        raise TrainingError("non-finite training loss")
    if require_gradient and not loss.requires_grad:
        raise TrainingError("training loss has no gradient path")
    return loss


def _finish_update(
    model: SyncModel,
    optimizer: torch.optim.Optimizer,
    pair_count: int,
    max_grad_norm: float,
    scaler: torch.amp.GradScaler | None,
) -> float:
    parameters = list(model.parameters())
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(pair_count)
    gradient_values = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradient_values or not any(bool(torch.count_nonzero(gradient).item()) for gradient in gradient_values):
        raise TrainingError("zero gradient before optimizer step")
    for branch_name in ("visual_encoder", "audio_encoder"):
        branch = getattr(model, branch_name, None)
        if branch is None:
            raise TrainingError(f"model is missing {branch_name}")
        branch_gradients = [parameter.grad for parameter in branch.parameters() if parameter.grad is not None]
        if not branch_gradients or not any(bool(torch.count_nonzero(gradient).item()) for gradient in branch_gradients):
            raise TrainingError(f"zero gradient in {branch_name}")
    if not all(bool(torch.isfinite(gradient).all().item()) for gradient in gradient_values):
        suffix = "" if scaler is None else f" (amp_scale={float(scaler.get_scale())})"
        raise TrainingError(f"non-finite gradient before optimizer step{suffix}")
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    if not bool(torch.isfinite(grad_norm).item()):
        raise TrainingError("non-finite gradient norm")
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.item())


@torch.no_grad()
def evaluate(model: SyncModel, tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor], device: torch.device) -> float:
    visual, audio, labels = tensors
    model.train(False)
    loss = cosine_bce_loss(model, visual.to(device), audio.to(device), labels.to(device), False, require_gradient=False)
    return float(loss.item()) / len(labels)


def train_epoch(
    model: SyncModel,
    optimizer: torch.optim.Optimizer,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: TrainState,
    config: TrainConfig,
    device: torch.device,
    *,
    max_updates: int | None = None,
    scaler: torch.amp.GradScaler | None = None,
    phase_recorder: ProfilePhaseRecorder | None = None,
) -> TrainOutcome:
    visual, audio, labels = tensors
    if len(visual) != len(audio) or len(visual) != len(labels):
        raise TrainingError("training tensor lengths differ")
    if state.next_example_cursor > len(labels):
        raise TrainingError("resume cursor exceeds epoch length")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    cursor = state.next_example_cursor
    updates = 0
    metrics: list[UpdateMetric] = []
    while cursor < len(labels):
        update_start = cursor
        loss_total = 0.0
        pair_count = 0
        update_end = min(len(labels), update_start + config.effective_batch_size)
        update_labels = labels[update_start:update_end]
        if len(update_labels) % 2 or float(update_labels.sum().item()) * 2 != len(update_labels):
            raise TrainingError("optimizer update is not exactly label-balanced")
        while cursor < update_end:
            with nullcontext() if phase_recorder is None else phase_recorder.measure("data_wait"):
                end = min(update_end, cursor + config.microbatch_size)
                visual_batch = visual[cursor:end]
                audio_batch = audio[cursor:end]
                label_batch = labels[cursor:end]
            with nullcontext() if phase_recorder is None else phase_recorder.measure("h2d"):
                visual_device = visual_batch.to(device)
                audio_device = audio_batch.to(device)
                label_device = label_batch.to(device)
            with nullcontext() if phase_recorder is None else phase_recorder.measure("forward_loss"):
                batch_loss = cosine_bce_loss(model, visual_device, audio_device, label_device, config.amp_fp16)
            with nullcontext() if phase_recorder is None else phase_recorder.measure("backward"):
                if scaler is None:
                    batch_loss.backward()
                else:
                    scaler.scale(batch_loss).backward()
            loss_total += float(batch_loss.detach().item())
            pair_count += end - cursor
            cursor = end
        with nullcontext() if phase_recorder is None else phase_recorder.measure("optimizer"):
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm = _finish_update(model, optimizer, pair_count, config.max_grad_norm, scaler)
            if scaler is not None:
                scaler.update()
        next_update = state.global_update + updates + 1
        metrics.append(UpdateMetric(next_update, loss_total / pair_count, pair_count, grad_norm))
        updates += 1
        if max_updates is not None and updates >= max_updates:
            break
    next_epoch = state.epoch + int(cursor == len(labels))
    next_cursor = 0 if cursor == len(labels) else cursor
    result = replace(
        state,
        epoch=next_epoch,
        next_example_cursor=next_cursor,
        global_update=state.global_update + updates,
        examples_seen=state.examples_seen + cursor - state.next_example_cursor,
    )
    return TrainOutcome(result, tuple(metrics))
