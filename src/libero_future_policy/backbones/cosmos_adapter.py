from __future__ import annotations

import glob
import importlib
import math
import os
import resource
import shlex
import site
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from libero_future_policy.backbones.visual_feature_extractor import SimpleVisualFeatureExtractor
from libero_future_policy.utils.shapes import expect_rank


@dataclass
class CosmosAdapterConfig:
    feature_dim: int
    use_external_cosmos: bool = False
    external_cosmos_module: str | None = None
    freeze_backbone: bool = False
    image_channels: int = 3
    cosmos_block_index: int = 0
    cosmos_model_size: str = "2B"
    cosmos_resolution: str = "720"
    cosmos_fps: int = 16
    cosmos_aspect_ratio: str = "16:9"
    cosmos_natten: bool = False
    cosmos_dit_path: str | None = None
    cosmos_checkpoints_root: str | None = None
    cosmos_auto_fetch_checkpoints: bool = True
    cosmos_default_prompt: str = ""
    cosmos_num_conditional_frames: int = 1
    cosmos_intermediate_pool: str = "mean"
    # T5-11b on GPU during pipeline init often causes OOM (host kills process, exit 137); offload keeps it on CPU until encode_prompt.
    cosmos_offload_text_encoder: bool = True
    cosmos_downcast_text_encoder: bool = True
    cosmos_verbose_load: bool = False


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _cosmos_should_log_verbose(config: CosmosAdapterConfig) -> bool:
    return bool(config.cosmos_verbose_load) or _env_truthy("HORIZON_COSMOS_DEBUG")


def _cosmos_load_log(config: CosmosAdapterConfig, msg: str) -> None:
    if _cosmos_should_log_verbose(config):
        print(f"[CosmosAdapter] {msg}", file=sys.stderr, flush=True)


def _process_rss_mb() -> float | None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux: kilobytes; macOS: bytes
        if sys.platform == "darwin":
            return float(rss) / (1024 * 1024)
        return float(rss) / 1024.0
    except Exception:
        return None


def _cuda_mem_line(device_index: int = 0) -> str:
    if not torch.cuda.is_available():
        return "cuda: n/a"
    parts = [
        f"alloc={torch.cuda.memory_allocated(device_index) / 1e9:.2f}GB",
        f"reserved={torch.cuda.memory_reserved(device_index) / 1e9:.2f}GB",
    ]
    try:
        free_b, total_b = torch.cuda.mem_get_info(device_index)
        parts.append(f"free={free_b / 1e9:.2f}GB")
        parts.append(f"total={total_b / 1e9:.2f}GB")
    except Exception:
        pass
    rss = _process_rss_mb()
    if rss is not None:
        parts.append(f"proc_max_rss~={rss:.0f}MB")
    return "cuda: " + ", ".join(parts)


def _ensure_transformer_engine_nvrtc_path() -> None:
    """Avoid Transformer Engine failing on ``ldconfig | grep libnvrtc`` when NVRTC is only in the venv.

    ``transformer_engine`` resolves ``libnvrtc`` under ``CUDA_HOME`` / ``CUDA_PATH``, then shells out to
    ``ldconfig``. Pip's ``nvidia-cuda-nvrtc`` installs under ``site-packages/nvidia/cuda_nvrtc/lib/``,
    which is often not registered in ``ldconfig``. Set ``CUDA_PATH`` to that prefix when needed.

    Set ``HORIZON_SKIP_NVRTC_BOOTSTRAP=1`` to disable. Non-Linux no-ops.
    """
    if os.environ.get("HORIZON_SKIP_NVRTC_BOOTSTRAP", "").lower() in ("1", "true", "yes"):
        return
    if sys.platform != "linux":
        return

    so = "so"
    cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
    hits = glob.glob(f"{cuda_root}/**/libnvrtc.{so}*", recursive=True)
    hits = [h for h in hits if "stub" not in h and "libnvrtc-builtins" not in h]
    if hits:
        return

    pip_nvrtc: Path | None = None
    for sp in site.getsitepackages():
        lib = Path(sp) / "nvidia" / "cuda_nvrtc" / "lib"
        if lib.is_dir() and any(lib.glob("libnvrtc.so*")):
            pip_nvrtc = lib.parent
            break
    if pip_nvrtc is None:
        us = site.getusersitepackages()
        if us:
            lib = Path(us) / "nvidia" / "cuda_nvrtc" / "lib"
            if lib.is_dir() and any(lib.glob("libnvrtc.so*")):
                pip_nvrtc = lib.parent
    if pip_nvrtc is None:
        return

    if os.environ.get("CUDA_HOME"):
        del os.environ["CUDA_HOME"]
    os.environ["CUDA_PATH"] = str(pip_nvrtc)


def _apply_cosmos_predict2_checkpoints_root(checkpoints_root: Path) -> None:
    """Point imaginaire at an absolute checkpoints tree before pipeline modules read paths."""
    os.environ["COSMOS_PREDICT2_ARGS"] = shlex.join(["--checkpoints", str(checkpoints_root.resolve())])
    for mod_name in ("imaginaire.constants", "cosmos_predict2.configs.base.config_video2world"):
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])


def _prepare_cosmos_checkpoints_directory(config: CosmosAdapterConfig) -> tuple[Path | None, str | None]:
    """Ensure ``ROOT/nvidia/Cosmos-Predict2-{size}-Video2World/`` and T5 under ``ROOT/google-t5/t5-11b``.

    Uses ``COSMOS_CHECKPOINTS_DIR``, ``cosmos_checkpoints_root``, or ``~/.cache/horizon/cosmos_checkpoints``.
    Optionally downloads from Hugging Face when ``cosmos_auto_fetch_checkpoints`` is True.

    Returns ``(root, None)`` on success, or ``(None, reason)`` when checkpoints cannot be prepared.
    """
    if config.cosmos_checkpoints_root:
        root = Path(config.cosmos_checkpoints_root).expanduser()
    elif os.environ.get("COSMOS_CHECKPOINTS_DIR"):
        root = Path(os.environ["COSMOS_CHECKPOINTS_DIR"]).expanduser()
    else:
        root = Path.home() / ".cache" / "horizon" / "cosmos_checkpoints"
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_size = config.cosmos_model_size
    repo_id = f"nvidia/Cosmos-Predict2-{model_size}-Video2World"
    video_dir = root / "nvidia" / f"Cosmos-Predict2-{model_size}-Video2World"
    natten_suffix = "-natten" if config.cosmos_natten else ""
    model_name = f"model-{config.cosmos_resolution}p-{config.cosmos_fps}fps{natten_suffix}.pt"
    model_path = video_dir / model_name
    tokenizer_path = video_dir / "tokenizer" / "tokenizer.pth"
    t5_dir = root / "google-t5" / "t5-11b"

    def video_ready() -> bool:
        return model_path.is_file() and tokenizer_path.is_file()

    def t5_ready() -> bool:
        return (t5_dir / "config.json").is_file()

    if not video_ready() or not t5_ready():
        if not config.cosmos_auto_fetch_checkpoints:
            reason = (
                "Checkpoints are missing and auto-download is disabled "
                f"(cosmos_auto_fetch_checkpoints=False). Expected Video2World weights at {model_path}, "
                f"tokenizer at {tokenizer_path}, and T5 config at {t5_dir / 'config.json'}. "
                "Enable auto-fetch, install huggingface_hub and download, or pass --checkpoints-root / set "
                "COSMOS_CHECKPOINTS_DIR to a tree that already contains these paths."
            )
            warnings.warn(reason, stacklevel=2)
            return None, reason
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            reason = (
                "Checkpoints are missing and huggingface_hub is not installed, so auto-download cannot run. "
                f"Install huggingface_hub or place weights manually: {model_path}, {tokenizer_path}, "
                f"{t5_dir / 'config.json'}."
            )
            warnings.warn(reason, stacklevel=2)
            return None, reason
        try:
            if not video_ready():
                video_dir.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(video_dir),
                    local_dir_use_symlinks=False,
                )
            if not t5_ready():
                t5_dir.parent.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id="google-t5/t5-11b",
                    local_dir=str(t5_dir),
                    local_dir_use_symlinks=False,
                )
        except Exception as exc:
            reason = f"Cosmos checkpoint download from Hugging Face failed ({type(exc).__name__}: {exc})."
            warnings.warn(reason, stacklevel=2)
            return None, reason

    if not video_ready() or not t5_ready():
        reason = (
            "Checkpoints are still incomplete after download or layout mismatch. "
            f"Expected {model_path}, {tokenizer_path}, and {t5_dir / 'config.json'}."
        )
        return None, reason
    return root, None


class CosmosAdapter(nn.Module):
    """Repository-local boundary around any external Cosmos runtime.

    The rest of the repo should only depend on this adapter and never import
    Cosmos internals directly.
    """

    cosmos_intermediate_proj: nn.Linear | None
    cosmos_pipeline_load_error: str | None

    def __init__(self, config: CosmosAdapterConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.runtime: Any | None = None
        self._cosmos_pipe: Any | None = None
        self.cosmos_pipeline_load_error = None
        self.encoder = SimpleVisualFeatureExtractor(
            in_channels=config.image_channels,
            feature_dim=config.feature_dim,
        )
        self.cosmos_intermediate_proj = None

        if config.use_external_cosmos:
            self.runtime = self._try_load_external_runtime(config.external_cosmos_module)
            self._load_cosmos_pipeline()

        if self.cosmos_intermediate_proj is None and config.cosmos_intermediate_pool == "mean":
            if self._cosmos_pipe is not None:
                d_model = int(self._cosmos_pipe.dit.model_channels)
                if d_model != config.feature_dim:
                    self.cosmos_intermediate_proj = nn.Linear(d_model, config.feature_dim)

        if config.freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            if self.cosmos_intermediate_proj is not None:
                for parameter in self.cosmos_intermediate_proj.parameters():
                    parameter.requires_grad = False
            if self._cosmos_pipe is not None:
                if self._cosmos_pipe.text_encoder is not None:
                    for parameter in self._cosmos_pipe.text_encoder.parameters():
                        parameter.requires_grad = False
                for parameter in self._cosmos_pipe.dit.parameters():
                    parameter.requires_grad = False

    def _try_load_external_runtime(self, module_name: str | None) -> Any | None:
        if module_name is None:
            return None
        try:
            return importlib.import_module(module_name)
        except Exception:
            return None

    def _load_cosmos_pipeline(self) -> None:
        self.cosmos_pipeline_load_error = None
        _ensure_transformer_engine_nvrtc_path()
        if not torch.cuda.is_available():
            self.cosmos_pipeline_load_error = (
                "CUDA is not available; the Video2World pipeline is only loaded when torch.cuda.is_available() is True."
            )
            self._cosmos_pipe = None
            return

        # Avoid multi-GB HF downloads when no GPU will run the pipeline.
        checkpoints_root, ckpt_error = _prepare_cosmos_checkpoints_directory(self.config)
        if checkpoints_root is None:
            self.cosmos_pipeline_load_error = ckpt_error or "Checkpoint directory could not be prepared (unknown reason)."
            self._cosmos_pipe = None
            return
        _apply_cosmos_predict2_checkpoints_root(checkpoints_root)

        try:
            import attrs
            from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
            from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
            from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint
        except ImportError as exc:
            self.cosmos_pipeline_load_error = (
                "Could not import the Cosmos Predict2 Python stack (cosmos_predict2, imaginaire, attrs, or a "
                f"transitive dependency). Install cosmos-predict2 in this environment. ImportError: {exc}"
            )
            self._cosmos_pipe = None
            return

        cfg = get_cosmos_predict2_video2world_pipeline(
            model_size=self.config.cosmos_model_size,
            resolution=self.config.cosmos_resolution,
            fps=self.config.cosmos_fps,
            natten=self.config.cosmos_natten,
        )
        cfg = attrs.evolve(
            cfg,
            guardrail_config=attrs.evolve(cfg.guardrail_config, enabled=False),
            prompt_refiner_config=attrs.evolve(cfg.prompt_refiner_config, enabled=False),
        )
        dit_path = self.config.cosmos_dit_path or get_cosmos_predict2_video2world_checkpoint(
            model_size=self.config.cosmos_model_size,
            resolution=self.config.cosmos_resolution,
            fps=self.config.cosmos_fps,
            aspect_ratio=self.config.cosmos_aspect_ratio,
            natten=self.config.cosmos_natten,
        )
        _cosmos_load_log(
            self.config,
            f"loading Video2WorldPipeline | offload_text_encoder={self.config.cosmos_offload_text_encoder} "
            f"downcast_text_encoder={self.config.cosmos_downcast_text_encoder} | dit_path={dit_path!r} | "
            f"{_cuda_mem_line()}",
        )
        try:
            _cosmos_load_log(self.config, "calling Video2WorldPipeline.from_config (after tokenizer: text encoder, then DiT)…")
            self._cosmos_pipe = Video2WorldPipeline.from_config(
                config=cfg,
                dit_path=dit_path,
                load_prompt_refiner=False,
                device="cuda",
                torch_dtype=torch.bfloat16,
                offload_text_encoder=self.config.cosmos_offload_text_encoder,
                downcast_text_encoder=self.config.cosmos_downcast_text_encoder,
            )
            _cosmos_load_log(self.config, f"Video2WorldPipeline ready | {_cuda_mem_line()}")
        except Exception as exc:
            _cosmos_load_log(
                self.config,
                f"Video2WorldPipeline.from_config failed ({type(exc).__name__}: {exc}) | {_cuda_mem_line()}",
            )
            self.cosmos_pipeline_load_error = (
                f"Video2WorldPipeline.from_config failed ({type(exc).__name__}: {exc}). "
                f"dit_path={dit_path!r}. Check GPU memory, checkpoint integrity, and that this resolution/fps/aspect "
                "match the weights; you can override weights with cosmos_dit_path / --dit-path."
            )
            warnings.warn(self.cosmos_pipeline_load_error, stacklevel=2)
            self._cosmos_pipe = None

    def _encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        expect_rank(frames, 5, "frames")
        return self.encoder(frames)

    def encode_current_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode_frames(frames)

    def encode_future_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return self._encode_frames(frames)

    def maybe_generate_future_features(
        self,
        batch: dict[str, Any],
        num_samples: int = 1,
        step: int | None = None,
    ) -> torch.Tensor | None:
        """TODO: connect frozen Cosmos Predict2 generation here.

        Expected return shape:
        - single sample: [B, T_future, D_vis]
        - multiple stochastic samples can later be [B, K, T_future, D_vis]
        """

        if self.runtime is None:
            return None

        # TODO: Replace this with real Cosmos rollout/generation calls.
        # The adapter contract intentionally hides all external repo details.
        return None

    def _cosmos_target_hw(self) -> tuple[int, int]:
        from cosmos_predict2.datasets.utils import VIDEO_RES_SIZE_INFO

        res = self.config.cosmos_resolution
        ar = self.config.cosmos_aspect_ratio
        sizes = VIDEO_RES_SIZE_INFO[res]
        if ar not in sizes:
            raise ValueError(f"Unknown aspect_ratio {ar!r} for resolution {res!r}, expected one of {list(sizes)}")
        h, w = sizes[ar]
        return h, w

    def _prepare_pixels_for_cosmos(self, frames: torch.Tensor) -> torch.Tensor:
        """Resize/crop to Cosmos pixel resolution and temporal length; return [B,C,T,H,W] float in [-1,1] on CUDA."""
        expect_rank(frames, 5, "frames")
        pipe = self._cosmos_pipe
        assert pipe is not None
        device = torch.device("cuda")
        target_h, target_w = self._cosmos_target_hw()
        expected_t = int(pipe.tokenizer.get_pixel_num_frames(pipe.config.state_t))

        x = frames
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif not torch.is_floating_point(x):
            x = x.float()
        if x.max() <= 1.0 + 1e-3 and x.min() >= -1.0e-3:
            x = x.clamp(0.0, 1.0)
            x = x * 2.0 - 1.0

        b, c, t, h, w = x.shape
        x = x.to(device=device, dtype=torch.float32)
        x_flat = rearrange(x, "b c t h w -> (b t) c h w")
        scale = max(target_w / w, target_h / h)
        new_h = int(math.ceil(scale * h))
        new_w = int(math.ceil(scale * w))
        x_flat = F.interpolate(x_flat, size=(new_h, new_w), mode="bilinear", align_corners=False)
        x_flat = self._center_crop_spatial(x_flat, (target_h, target_w))
        x = rearrange(x_flat, "(b t) c h w -> b c t h w", b=b, t=t)

        if t < expected_t:
            pad = expected_t - t
            last = x[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
            x = torch.cat([x, last], dim=2)
        elif t > expected_t:
            start = (t - expected_t) // 2
            x = x[:, :, start : start + expected_t, :, :]

        return x

    @staticmethod
    def _center_crop_spatial(x: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
        th, tw = size_hw
        _, _, h, w = x.shape
        i = max(0, (h - th) // 2)
        j = max(0, (w - tw) // 2)
        return x[:, :, i : i + th, j : j + tw]

    def _build_cosmos_data_batch(self, video_bc_thw: torch.Tensor, prompt: str) -> dict[str, Any]:
        from cosmos_predict2.pipelines.video2world import IS_PREPROCESSED_KEY, NUM_CONDITIONAL_FRAMES_KEY

        pipe = self._cosmos_pipe
        assert pipe is not None
        b, _, t, h, w = video_bc_thw.shape
        effective = prompt if prompt else self.config.cosmos_default_prompt
        prompts = [effective] * b
        _cosmos_load_log(self.config, f"encode_prompt (T5) start | {_cuda_mem_line()}")
        emb = pipe.encode_prompt(prompts).to(dtype=pipe.torch_dtype)
        _cosmos_load_log(self.config, f"encode_prompt (T5) done | {_cuda_mem_line()}")
        data_batch: dict[str, Any] = {
            "dataset_name": "video_data",
            pipe.input_video_key: video_bc_thw,
            "t5_text_embeddings": emb,
            "fps": torch.randint(16, 32, (b,), device=video_bc_thw.device),
            "padding_mask": torch.zeros(b, 1, h, w, device=video_bc_thw.device, dtype=pipe.torch_dtype),
            NUM_CONDITIONAL_FRAMES_KEY: self.config.cosmos_num_conditional_frames,
            IS_PREPROCESSED_KEY: True,
        }
        return data_batch

    def _sample_sigma_bt(
        self, batch_size: int, device: torch.device, denoise_level: float | None
    ) -> torch.Tensor:
        pipe = self._cosmos_pipe
        assert pipe is not None
        if denoise_level is None:
            sigma_b = pipe.scheduler.sample_sigma(batch_size).to(device=device, dtype=torch.float32)
        else:
            level = float(denoise_level)
            level = max(0.0, min(1.0, level))
            scfg = pipe.scheduler.config
            sig = scfg.sigma_min + level * (scfg.sigma_max - scfg.sigma_min)
            sigma_b = torch.full((batch_size,), sig, device=device, dtype=torch.float32)
        sigma_bt = rearrange(sigma_b, "b -> b 1")
        if pipe.config.adjust_video_noise:
            sigma_bt = sigma_bt * math.sqrt(float(pipe.config.state_t))
        return sigma_bt

    def maybe_extract_intermediate_features(
        self,
        frames: torch.Tensor,
        denoise_level: float | None = None,
        *,
        prompt: str | None = None,
    ) -> torch.Tensor | None:
        """Run one Video2World denoise step and return activations after DiT block k.

        Requires CUDA, ``use_external_cosmos=True``, and a successful Cosmos pipeline load.
        ``frames`` is ``[B, C, T, H, W]`` (dataset range ``[0,1]`` or ``[-1,1]`` float, or uint8).

        Returns:
            If ``cosmos_intermediate_pool`` is ``\"mean\"``: ``[B, feature_dim]`` (projected if needed).
            If ``\"none\"``: ``[B, T', H', W', D]`` patch-grid tensor after block k.
            ``None`` when Cosmos is unavailable or inputs are not on CUDA.
        """

        if self._cosmos_pipe is None or not torch.cuda.is_available():
            return None
        if frames.device.type != "cuda":
            return None

        pipe = self._cosmos_pipe
        n_blocks = len(pipe.dit.blocks)
        if n_blocks == 0:
            return None
        k = min(max(0, int(self.config.cosmos_block_index)), n_blocks - 1)

        text_prompt = self.config.cosmos_default_prompt if prompt is None else prompt

        captured: list[torch.Tensor] = []

        def _hook(_module: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            captured.append(out.detach())

        hook_handle = pipe.dit.blocks[k].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                _cosmos_load_log(self.config, f"maybe_extract_intermediate_features: prepare pixels | {_cuda_mem_line()}")
                video = self._prepare_pixels_for_cosmos(frames)
                _cosmos_load_log(self.config, f"maybe_extract_intermediate_features: build batch | {_cuda_mem_line()}")
                data_batch = self._build_cosmos_data_batch(video, text_prompt)
                _, x0, condition = pipe.get_data_and_condition(data_batch)
                _cosmos_load_log(self.config, f"maybe_extract_intermediate_features: denoise | {_cuda_mem_line()}")
                b = x0.shape[0]
                sigma_bt = self._sample_sigma_bt(b, x0.device, denoise_level)
                epsilon = torch.randn_like(x0)
                xt = x0 + epsilon * rearrange(sigma_bt.to(dtype=x0.dtype), "b t -> b 1 t 1 1")
                pipe.denoise(xt, sigma_bt.to(dtype=x0.dtype), condition, use_cuda_graphs=False)
                _cosmos_load_log(self.config, f"maybe_extract_intermediate_features: denoise done | {_cuda_mem_line()}")
        finally:
            hook_handle.remove()

        if not captured:
            return None
        h_out = captured[-1]
        pool = self.config.cosmos_intermediate_pool
        if pool == "none":
            return h_out
        if pool != "mean":
            raise ValueError(f"Unknown cosmos_intermediate_pool {pool!r}, expected 'mean' or 'none'")
        pooled = h_out.mean(dim=(1, 2, 3))
        if self.cosmos_intermediate_proj is not None:
            return self.cosmos_intermediate_proj(pooled)
        return pooled

    def empty_future_features(
        self,
        batch_size: int,
        future_window: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, future_window, self.feature_dim, device=device)
