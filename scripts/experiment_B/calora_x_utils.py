"""Compact representative-gradient memory and soft projection for CaLoRA-X.

This module is intentionally independent from ``calora_utils.py`` so the
primary CaLoRA implementation remains unchanged.  It contains no replay data:
only compressed LoRA-gradient bases and coefficient centroids are persisted.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F


MEMORY_VERSION = 1


def make_snapshot_schedule(total_steps: int, samples: int, start_fraction: float) -> list[int]:
    """Return fixed optimiser-boundary positions at which to collect gradients."""
    if total_steps < 1:
        raise ValueError("CaLoRA-X needs a positive number of optimiser steps.")
    if not 1 <= samples <= total_steps:
        raise ValueError(f"samples must be in [1, {total_steps}], got {samples}.")
    if not 0.0 <= start_fraction <= 1.0:
        raise ValueError("start_fraction must be in [0, 1].")

    start = max(1, math.ceil(total_steps * start_fraction))
    if samples == 1:
        return [total_steps]
    raw = [round(start + i * (total_steps - start) / (samples - 1)) for i in range(samples)]
    schedule = sorted(set(max(1, min(total_steps, int(step))) for step in raw))
    # Very short tasks can round positions together. Fill deterministic gaps.
    candidate = start
    while len(schedule) < samples:
        if candidate not in schedule:
            schedule.append(candidate)
        candidate += 1
        if candidate > total_steps:
            break
    return sorted(schedule)


class GradientSnapshotCollector:
    """Keeps a small number of detached LoRA gradients in host RAM only."""

    def __init__(
        self,
        parameter_names: list[str],
        total_steps: int,
        samples: int = 4,
        start_fraction: float = 0.4,
    ) -> None:
        self.parameter_names = list(parameter_names)
        self.schedule = make_snapshot_schedule(total_steps, samples, start_fraction)
        self.snapshots: list[list[torch.Tensor]] = [[] for _ in parameter_names]
        self.boundary_count = 0
        self.next_schedule_index = 0

    @property
    def captured(self) -> int:
        return self.next_schedule_index

    def maybe_capture(self, gradients: Iterable[torch.Tensor]) -> bool:
        """Capture one full LoRA-gradient set if this is a scheduled boundary."""
        self.boundary_count += 1
        if self.next_schedule_index >= len(self.schedule):
            return False
        if self.boundary_count < self.schedule[self.next_schedule_index]:
            return False

        gradients = list(gradients)
        if len(gradients) != len(self.snapshots):
            raise ValueError("CaLoRA-X snapshot tensor count does not match the LoRA parameter list.")
        for bucket, gradient in zip(self.snapshots, gradients):
            if not torch.isfinite(gradient).all():
                raise FloatingPointError("CaLoRA-X refused to store a non-finite gradient snapshot.")
            bucket.append(gradient.detach().to(device="cpu", dtype=torch.float16).clone())
        self.next_schedule_index += 1
        return True

    def clear(self) -> None:
        self.snapshots = [[] for _ in self.parameter_names]


def _normalise_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float()
    return tensor / (torch.linalg.vector_norm(tensor) + 1e-8)


@torch.no_grad()
def build_representative_memory(
    collector: GradientSnapshotCollector,
    parameter_shapes: list[tuple[int, ...]],
    memory_rank: int,
    work_device: torch.device,
) -> dict[str, Any]:
    """Compress temporal gradient snapshots to a rank-limited basis per tensor.

    Snapshots are already on CPU.  Only one concatenated matrix is moved to the
    work device at once, keeping the task-end operation bounded in VRAM.
    """
    if collector.captured != len(collector.schedule):
        raise RuntimeError(
            f"CaLoRA-X collected {collector.captured}/{len(collector.schedule)} scheduled snapshots; "
            "a task ended before the memory schedule completed."
        )
    if len(parameter_shapes) != len(collector.snapshots):
        raise ValueError("CaLoRA-X parameter shape count does not match collected snapshots.")
    if memory_rank < 1:
        raise ValueError("memory_rank must be positive.")

    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for index, (snapshots, expected_shape) in enumerate(zip(collector.snapshots, parameter_shapes)):
        if not snapshots:
            raise RuntimeError(f"No CaLoRA-X snapshots were retained for tensor {index}.")
        if any(tuple(snapshot.shape) != tuple(expected_shape) for snapshot in snapshots):
            raise ValueError(f"Snapshot shape mismatch for CaLoRA-X tensor {index}.")
        if len(expected_shape) not in (1, 2):
            raise ValueError(f"CaLoRA-X supports only 1D/2D LoRA tensors, got {expected_shape}.")

        normalised = [_normalise_snapshot(snapshot) for snapshot in snapshots]
        if len(expected_shape) == 1:
            matrix_cpu = torch.stack(normalised, dim=1)  # d x snapshots
            mean_tensor_cpu = torch.stack(normalised, dim=0).mean(dim=0)
            kind = "vector"
        else:
            matrix_cpu = torch.cat(normalised, dim=1)  # rows x (columns * snapshots)
            mean_tensor_cpu = torch.stack(normalised, dim=0).mean(dim=0)
            kind = "matrix"

        matrix = matrix_cpu.to(work_device, dtype=torch.float32)
        q = min(memory_rank, matrix.shape[0], matrix.shape[1])
        # Randomized PCA is sufficient for the deliberately small basis and is
        # far cheaper than a full SVD across 576 LoRA tensors.
        basis, singular_values, _ = torch.pca_lowrank(matrix, q=q, center=False, niter=2)
        importance = (singular_values / (singular_values.max() + 1e-8)).clamp(0.0, 1.0)
        mean_tensor = mean_tensor_cpu.to(work_device, dtype=torch.float32)
        mean_coeff = basis.transpose(0, 1) @ mean_tensor

        entry = {
            "kind": kind,
            "shape": tuple(expected_shape),
            "basis": basis.to(device="cpu", dtype=torch.float16).contiguous(),
            "importance": importance.to(device="cpu", dtype=torch.float32).contiguous(),
            "mean_coeff": mean_coeff.to(device="cpu", dtype=torch.float16).contiguous(),
        }
        total_bytes += sum(value.numel() * value.element_size() for value in entry.values() if torch.is_tensor(value))
        entries.append(entry)
        del matrix, matrix_cpu, mean_tensor, mean_tensor_cpu, basis, singular_values, importance, mean_coeff

    return {
        "version": MEMORY_VERSION,
        "parameter_names": collector.parameter_names,
        "parameter_shapes": [tuple(shape) for shape in parameter_shapes],
        "snapshot_schedule": collector.schedule,
        "snapshot_count": collector.captured,
        "memory_rank": memory_rank,
        "entries": entries,
        "storage_bytes": total_bytes,
    }


def save_representative_memory(memory: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(memory, path)


@torch.no_grad()
def load_representative_memory(
    path: Path,
    expected_names: list[str],
    expected_shapes: list[tuple[int, ...]],
    device: torch.device,
) -> dict[str, Any]:
    """Load and exhaustively validate one compact task memory."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CaLoRA-X representative memory: {path}")
    memory = torch.load(path, map_location="cpu", weights_only=True)
    if memory.get("version") != MEMORY_VERSION:
        raise ValueError(f"Unsupported CaLoRA-X memory version in {path}.")
    if memory.get("parameter_names") != expected_names:
        raise ValueError(f"CaLoRA-X parameter order mismatch in {path}; start a fresh Q/K/V/O run.")
    if [tuple(shape) for shape in memory.get("parameter_shapes", [])] != [tuple(shape) for shape in expected_shapes]:
        raise ValueError(f"CaLoRA-X parameter shapes mismatch in {path}; start a fresh Q/K/V/O run.")
    entries = memory.get("entries", [])
    if len(entries) != len(expected_names):
        raise ValueError(f"CaLoRA-X tensor count mismatch in {path}.")

    loaded_entries = []
    for index, (entry, shape) in enumerate(zip(entries, expected_shapes)):
        basis = entry.get("basis")
        importance = entry.get("importance")
        mean_coeff = entry.get("mean_coeff")
        if not all(torch.is_tensor(value) for value in (basis, importance, mean_coeff)):
            raise ValueError(f"Invalid CaLoRA-X tensors at index {index} in {path}.")
        if tuple(entry.get("shape", ())) != tuple(shape) or basis.ndim != 2:
            raise ValueError(f"Invalid CaLoRA-X basis shape at index {index} in {path}.")
        if basis.shape[0] != shape[0] or basis.shape[1] != importance.numel() or mean_coeff.shape[0] != basis.shape[1]:
            raise ValueError(f"Incompatible CaLoRA-X basis dimensions at index {index} in {path}.")
        if not all(torch.isfinite(value).all() for value in (basis, importance, mean_coeff)):
            raise FloatingPointError(f"Non-finite CaLoRA-X memory at index {index} in {path}.")
        if (importance < 0).any() or (importance > 1).any():
            raise ValueError(f"CaLoRA-X importance outside [0, 1] at index {index} in {path}.")
        loaded_entries.append({
            "kind": entry["kind"],
            "shape": tuple(shape),
            "basis": basis.to(device=device, dtype=torch.float32),
            "importance": importance.to(device=device, dtype=torch.float32),
            "mean_coeff": mean_coeff.to(device=device, dtype=torch.float32),
        })
    return {**memory, "entries": loaded_entries}


@torch.no_grad()
def soft_project_gradient(
    gradient: torch.Tensor,
    entry: dict[str, Any],
    attenuation: float,
    min_scale: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply one prior task's continuous, conflict-gated soft projection."""
    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("attenuation must be in [0, 1].")
    if not 0.0 < min_scale <= 1.0:
        raise ValueError("min_scale must be in (0, 1].")
    if tuple(gradient.shape) != tuple(entry["shape"]):
        raise ValueError("Current gradient shape does not match CaLoRA-X memory.")

    original_dtype = gradient.dtype
    gradient_f = gradient.detach().float()
    basis = entry["basis"].to(gradient_f.device)
    coeff = basis.transpose(0, 1) @ gradient_f
    projected = basis @ coeff
    orthogonal = gradient_f - projected
    mean_coeff = entry["mean_coeff"].to(gradient_f.device)
    cosine = F.cosine_similarity(coeff.reshape(1, -1), mean_coeff.reshape(1, -1), dim=1, eps=1e-8)[0]
    conflict = (-cosine).clamp(0.0, 1.0)
    scales = (1.0 - attenuation * conflict * entry["importance"].to(gradient_f.device)).clamp(min_scale, 1.0)
    scale_shape = (scales.numel(),) + (1,) * (coeff.ndim - 1)
    corrected = orthogonal + basis @ (scales.reshape(scale_shape) * coeff)

    projection_ratio = (torch.linalg.vector_norm(projected) / (torch.linalg.vector_norm(gradient_f) + 1e-8)).clamp(0.0, 1.0)
    return corrected.to(dtype=original_dtype), {
        "projection_ratio": float(projection_ratio.item()),
        "cosine": float(cosine.item()),
        "conflict": float(conflict.item()),
        "mean_scale": float(scales.mean().item()),
    }


def run_toy_projection_check() -> None:
    """Cheap deterministic invariant check used by the training script's self-test."""
    entry = {
        "shape": (2,),
        "basis": torch.tensor([[1.0], [0.0]]),
        "importance": torch.tensor([1.0]),
        "mean_coeff": torch.tensor([1.0]),
    }
    aligned, aligned_stats = soft_project_gradient(torch.tensor([2.0, 3.0]), entry, 0.5, 0.1)
    conflicting, conflict_stats = soft_project_gradient(torch.tensor([-2.0, 3.0]), entry, 0.5, 0.1)
    if not torch.allclose(aligned, torch.tensor([2.0, 3.0])) or aligned_stats["conflict"] != 0.0:
        raise AssertionError("CaLoRA-X aligned-gradient invariant failed.")
    if not (abs(conflicting[1].item() - 3.0) < 1e-6 and abs(conflicting[0].item()) < 2.0 and conflict_stats["conflict"] > 0):
        raise AssertionError("CaLoRA-X conflict attenuation invariant failed.")
