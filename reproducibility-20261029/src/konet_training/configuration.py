from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import assert_never

from .adapter_config import LoRAConfig
from .errors import ConfigurationError


GIB = 1024**3


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["toy", "stable_syncnet"]
    feature_dim: int = 2
    vendor_root: Path | None = None
    architecture_config: Path | None = None
    base_checkpoint: Path | None = None
    lora: LoRAConfig | None = None

    @model_validator(mode="after")
    def stable_paths_are_complete(self) -> ModelConfig:
        if self.lora is not None and self.kind != "stable_syncnet":
            raise ConfigurationError("LoRA requires the StableSyncNet attention architecture")
        if self.kind == "stable_syncnet" and (
            self.vendor_root is None or self.architecture_config is None or self.base_checkpoint is None
        ):
            raise ConfigurationError("stable_syncnet requires vendor_root, architecture_config, and base_checkpoint")
        return self


class FixtureDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["tensor_fixture"]
    train_bundle: Path
    validation_bundle: Path


class Frozen40000CohortConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["frozen40000_epoch0"]
    canonical_epoch: Literal[0]
    canonical_start_ordinal: Literal[6912]
    example_count: Literal[40000]
    historical_examples_consumed: Literal[6912]
    source_checkpoint_epoch: Literal[0]
    source_checkpoint_cursor: Literal[6912]
    source_checkpoint_global_update: Literal[54]
    source_checkpoint_examples_seen: Literal[6912]
    source_config_sha256: Literal["991c9b7a69db2f6f4195bfe6854f22bd8d5e55ecfd7ec8cc14bbd9ea9ddb3632"]


class NativeDataConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    kind: Literal["native_pairs"]
    data_root: Path
    inventory_path: Path
    prepared_path: Path
    cache_root: Path
    cache_gib: float = Field(default=12.0, gt=0, le=12.0)
    native_vendor_root: Path
    train_seed: int = 1701
    offline_training_root: Path | None = None
    frozen_training_cohort: Frozen40000CohortConfig | None = None
    transfer_plan_path: Path | None = None
    transfer_selected_path: Path | None = None
    transfer_root: Path | None = None
    local_media_manifest_path: Path | None = None
    local_media_root: Path | None = None
    component_cache_root: Path | None = None
    component_cache_gib: float | None = Field(default=None, gt=0, le=4.0)
    geometry_cache_backend: Literal["sqlite", "compact"] = "sqlite"
    geometry_cache_path: Path | None = None
    compact_geometry_root: Path | None = None
    geometry_cache_gib: float | None = Field(default=None, gt=0, le=4.0)

    @model_validator(mode="after")
    def offline_root_is_absolute(self) -> NativeDataConfig:
        if self.offline_training_root is not None and not self.offline_training_root.is_absolute():
            raise ConfigurationError("offline_training_root must be absolute")
        if self.frozen_training_cohort is not None and self.offline_training_root is None:
            raise ConfigurationError("frozen_training_cohort requires offline_training_root")
        return self

    @model_validator(mode="after")
    def transfer_paths_are_complete(self) -> NativeDataConfig:
        supplied = (
            self.transfer_plan_path is not None,
            self.transfer_selected_path is not None,
            self.transfer_root is not None,
        )
        if any(supplied) and not all(supplied):
            raise ConfigurationError("transfer_plan_path, transfer_selected_path, and transfer_root must be set together")
        local_supplied = (self.local_media_manifest_path is not None, self.local_media_root is not None)
        if any(local_supplied) and not all(local_supplied):
            raise ConfigurationError("local_media_manifest_path and local_media_root must be set together")
        if all(local_supplied) and all(supplied):
            raise ConfigurationError("local media and transfer media are mutually exclusive")
        return self

    @model_validator(mode="after")
    def component_cache_is_bounded(self) -> NativeDataConfig:
        if (self.component_cache_root is None) != (self.component_cache_gib is None):
            raise ConfigurationError("component_cache_root and component_cache_gib must be set together")
        match self.geometry_cache_backend:
            case "sqlite":
                if self.compact_geometry_root is not None:
                    raise ConfigurationError("compact geometry root requires compact backend")
                geometry_supplied = (self.geometry_cache_path is not None, self.geometry_cache_gib is not None)
                if any(geometry_supplied) and not all(geometry_supplied):
                    raise ConfigurationError("geometry_cache_path and geometry_cache_gib must be set together")
            case "compact":
                if self.geometry_cache_path is not None:
                    raise ConfigurationError("compact geometry backend is mutually exclusive with SQLite path")
                if self.compact_geometry_root is None or self.geometry_cache_gib is None:
                    raise ConfigurationError("compact geometry root and geometry_cache_gib must be set together")
            case unreachable:
                assert_never(unreachable)
        persistent_gib = (self.component_cache_gib or 0.0) + (self.geometry_cache_gib or 0.0)
        if persistent_gib + 0.25 >= self.cache_gib:
            if self.geometry_cache_gib is None:
                raise ConfigurationError("component cache leaves insufficient recording capacity")
            raise ConfigurationError("persistent caches leave insufficient recording capacity")
        return self


class OptimizationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    epochs: int = Field(ge=1, le=65)
    microbatch_size: int = Field(ge=1, le=128)
    effective_batch_size: int = Field(default=128, ge=2)
    learning_rate: float = Field(default=5e-6, gt=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    amp_fp16: bool = True
    loader_workers: int = Field(default=4, ge=2, le=12)

    @model_validator(mode="after")
    def loader_worker_grid(self) -> OptimizationConfig:
        if self.loader_workers not in (2, 4, 8, 12):
            raise ConfigurationError("loader_workers must be 2, 4, 8, or 12")
        return self

    @model_validator(mode="after")
    def effective_batch_is_balanced(self) -> OptimizationConfig:
        if self.effective_batch_size % 2:
            raise ConfigurationError("effective_batch_size must be even")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    profile_protocol: Literal["full_v1", "reduced_v1"] = "full_v1"
    profile_forecast_seconds: float = Field(default=2_700, gt=0, le=3_600)
    reduced_profile_stage_cap_seconds: float = Field(default=3_600, gt=0, le=4_800)
    reduced_profile_termination_seconds: float = Field(default=300, gt=0, le=300)
    overfit_forecast_seconds: float = Field(default=2_700, gt=0, le=2_700)
    train_forecast_seconds: float = Field(default=7_200, gt=0)
    validation_forecast_seconds: float = Field(default=3_600, gt=0)
    final_evaluation_seconds: float = Field(default=7_200, ge=7_200)
    save_seconds: float = Field(default=300, ge=300)
    cache_growth_bytes: int = Field(default=0, ge=0)
    output_growth_bytes: int = Field(default=28 * 1024**3, ge=0, le=35 * 1024**3)
    heartbeat_max_age_seconds: float = Field(default=300, gt=0)
    profile_microbatch_grid: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    profile_worker_grid: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def full_profile_uses_legacy_forecast_limit(self) -> RuntimeConfig:
        if self.profile_protocol == "full_v1" and self.profile_forecast_seconds > 2_700:
            raise ConfigurationError("full_v1 profile forecast cannot exceed 2700 seconds")
        return self

    @model_validator(mode="after")
    def profile_grid_is_ordered_and_bounded(self) -> RuntimeConfig:
        if not self.profile_microbatch_grid or tuple(sorted(set(self.profile_microbatch_grid))) != self.profile_microbatch_grid:
            raise ConfigurationError("profile_microbatch_grid must be strictly increasing")
        if any(value not in (1, 2, 4, 8, 16, 32) for value in self.profile_microbatch_grid):
            raise ConfigurationError("profile_microbatch_grid contains an unsupported size")
        if self.profile_worker_grid is not None:
            if not self.profile_worker_grid or tuple(sorted(set(self.profile_worker_grid))) != self.profile_worker_grid:
                raise ConfigurationError("profile_worker_grid must be strictly increasing")
            if any(value not in (4, 8, 12) for value in self.profile_worker_grid):
                raise ConfigurationError("profile_worker_grid contains an unsupported count")
        return self


class StorageRelocationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["nvme_relocation_v1"]
    source_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_run_root: Path
    volume_root: Path
    minimum_free_bytes: int = Field(ge=25 * GIB)
    protected_gpu_reserve_seconds: Literal[15_000]

    @model_validator(mode="after")
    def roots_are_absolute_and_distinct(self) -> StorageRelocationConfig:
        if not self.source_run_root.is_absolute() or not self.volume_root.is_absolute():
            raise ConfigurationError("storage relocation roots must be absolute")
        return self

class RunConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: Literal[1]
    run_root: Path
    device: Literal["cpu", "cuda"] = "cuda"
    seed: int
    model: ModelConfig
    data: Annotated[FixtureDataConfig | NativeDataConfig, Field(discriminator="kind")]
    training: OptimizationConfig
    runtime: RuntimeConfig = RuntimeConfig()
    storage: StorageRelocationConfig | None = None

    @model_validator(mode="after")
    def fixture_mode_is_toy_only(self) -> RunConfig:
        if self.model.lora is None and self.training.epochs > 10:
            raise ConfigurationError("epochs above 10 require LoRA")
        if self.model.lora is None and self.training.microbatch_size > 32:
            raise ConfigurationError("microbatches above 32 require LoRA")
        if self.model.kind == "stable_syncnet" and self.data.kind != "native_pairs":
            raise ConfigurationError("stable_syncnet requires native_pairs data")
        if self.model.kind == "stable_syncnet" and self.model.lora is None and self.training.learning_rate != 5e-6:
            raise ConfigurationError("stable_syncnet learning_rate must be 5e-6")
        if self.model.lora is not None and self.training.learning_rate > 1e-3:
            raise ConfigurationError("LoRA learning_rate must not exceed 1e-3")
        if self.model.kind == "stable_syncnet" and self.training.effective_batch_size != 128:
            raise ConfigurationError("stable_syncnet effective_batch_size must be 128")
        if self.model.kind == "stable_syncnet" and not self.training.amp_fp16:
            raise ConfigurationError("stable_syncnet requires AMP FP16")
        if self.storage is not None:
            volume_root = self.storage.volume_root.resolve()
            run_root = self.run_root.resolve()
            if run_root == volume_root or volume_root not in run_root.parents:
                raise ConfigurationError("run_root must be contained by the approved NVMe volume")
        return self


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate_json(path.read_text(encoding="utf-8"))
