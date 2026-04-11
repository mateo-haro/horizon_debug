"""Cosmos Predict2 Video2World integration. Config, scripts, and examples: see ``README.md`` in this directory."""

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

from horizon.backbones.visual_feature_extractor import SimpleVisualFeatureExtractor
from horizon.utils.shapes import expect_rank


@dataclass
class CosmosAdapterConfig:
    feature_dim: int
    use_external_cosmos: bool = False
    external_cosmos_module: str | None = None
    freeze_backbone: bool = False
    image_channels: int = 3
    cosmos_block_index: int = 0
    cosmos_model_size: str = "2B"
    cosmos_resolution: str = "480"
    cosmos_fps: int = 16
    cosmos_aspect_ratio: str = "1:1"
    cosmos_natten: bool = False
    cosmos_dit_path: str | None = None
    cosmos_checkpoints_root: str | None = None
    cosmos_auto_fetch_checkpoints: bool = True
    cosmos_default_prompt: str = ""
    cosmos_num_conditional_frames: int = 5
    # Video2World denoise: "frame_replace" injects clean latents for the first N conditional frames; see ConditioningStrategy.
    cosmos_conditioning_strategy: str = "frame_replace"
    cosmos_intermediate_pool: str = "none"
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


class CosmosVideoAdapter(nn.Module):
    """Video2World pipeline wrapper (``CosmosAdapterConfig``). For policy training, see ``CosmosAdapter``."""

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
                    _p0 = next(self._cosmos_pipe.dit.parameters())
                    self.cosmos_intermediate_proj = nn.Linear(d_model, config.feature_dim).to(
                        device=_p0.device, dtype=_p0.dtype
                    )

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
            conditioning_strategy=self.config.cosmos_conditioning_strategy,
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
        # CPU-offloaded T5: Video2WorldPipeline.encode_prompt moves weights to CUDA, but
        # CosmosT5TextEncoder.encode_prompts still does .to(self.device) with the init-time
        # "cpu" string. Sync the string so token ids land on CUDA (no cosmos-predict2 patch).
        te = pipe.text_encoder
        if te is not None and hasattr(te, "device"):
            target = str(pipe.tensor_kwargs.get("device", "cuda"))
            if isinstance(te.device, str) and any(p.device.type == "cpu" for p in te.parameters()):
                te.device = target
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
        return_metadata: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]] | None:
        """Run one Video2World denoise step and return activations after DiT block k.

        Requires CUDA, ``use_external_cosmos=True``, and a successful Cosmos pipeline load.
        ``frames`` is ``[B, C, T, H, W]`` (dataset range ``[0,1]`` or ``[-1,1]`` float, or uint8).

        Returns:
            If ``return_metadata`` is False: same as before — tensor or ``None``.
            If ``return_metadata`` is True: ``(tensor, meta)`` or ``None``. ``meta`` includes
            ``prompt`` (str), ``denoise_level`` (``float | None``, argument passed in), and
            ``sigma_b`` (``torch.Tensor`` on CPU, shape ``[B]``, effective noise scale per batch item
            after ``_sample_sigma_bt``, including ``adjust_video_noise`` scaling).

            If ``cosmos_intermediate_pool`` is ``\"mean\"``: feature tensor is ``[B, feature_dim]`` (projected if needed).
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
        sigma_bt_used: torch.Tensor | None = None

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
                sigma_bt_used = sigma_bt
                epsilon = torch.randn_like(x0)
                xt = x0 + epsilon * rearrange(sigma_bt.to(dtype=x0.dtype), "b t -> b 1 t 1 1")
                pipe.denoise(xt, sigma_bt.to(dtype=x0.dtype), condition, use_cuda_graphs=False)
                _cosmos_load_log(self.config, f"maybe_extract_intermediate_features: denoise done | {_cuda_mem_line()}")
        finally:
            hook_handle.remove()

        if not captured or sigma_bt_used is None:
            return None
        h_out = captured[-1]
        pool = self.config.cosmos_intermediate_pool
        if pool == "none":
            feats = h_out
        elif pool == "mean":
            pooled = h_out.mean(dim=(1, 2, 3))
            if self.cosmos_intermediate_proj is not None:
                w = self.cosmos_intermediate_proj.weight
                if pooled.dtype != w.dtype:
                    pooled = pooled.to(dtype=w.dtype)
                feats = self.cosmos_intermediate_proj(pooled)
            else:
                feats = pooled
        else:
            raise ValueError(f"Unknown cosmos_intermediate_pool {pool!r}, expected 'mean' or 'none'")

        if return_metadata:
            sigma_b_cpu = sigma_bt_used.squeeze(-1).detach().cpu().float()
            meta: dict[str, Any] = {
                "prompt": text_prompt,
                "denoise_level": denoise_level,
                "sigma_b": sigma_b_cpu,
            }
            return feats, meta
        return feats

    def empty_future_features(
        self,
        batch_size: int,
        future_window: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.zeros(batch_size, future_window, self.feature_dim, device=device)


@dataclass
class CosmosDenoiseState:
    encoded_features: torch.Tensor
    hidden_states: list[torch.Tensor]
    tau: float
    noise_level: float
    sigma: float | None = None


@dataclass
class _PolicyCosmosEncodeOutput:
    latent: torch.Tensor
    tokens: torch.Tensor


@dataclass
class _PolicyCosmosRuntimeDenoiseOutput:
    encoded_tokens: torch.Tensor
    hidden_tokens: list[torch.Tensor]
    sigma: float


class _CosmosPredict2Runtime:
    """Thin runtime wrapper around the local Cosmos Predict2 LIBERO fork."""

    def __init__(
        self,
        repo_path: str | Path,
        device: str,
        torch_dtype: torch.dtype,
        model_size: str,
        resolution: str,
        fps: int,
        lora_checkpoint: str | None,
        lora_rank: int,
        lora_alpha: int,
        lora_target_modules: str,
        prompt_refiner_enabled: bool,
        guardrail_enabled: bool,
    ) -> None:
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("Cosmos Predict2 runtime requires CUDA for practical use.")

        if repo_path is not None:
            repo_path = Path(repo_path).resolve()
            if repo_path.exists():
                if str(repo_path) not in sys.path:
                    sys.path.insert(0, str(repo_path))
        try:
            import importlib

            importlib.import_module("cosmos_predict2")
        except ImportError as exc:
            hint = (
                " Install cosmos_predict2 (e.g. uv/pip) or pass an existing --repo-path to a checkout."
            )
            if repo_path is not None and not Path(repo_path).exists():
                hint = f" Repo path does not exist: {repo_path}.{hint}"
            raise FileNotFoundError(f"Cannot import cosmos_predict2.{hint}") from exc

        from cosmos_predict2.configs.base.config_video2world import get_cosmos_predict2_video2world_pipeline
        from cosmos_predict2.conditioner import DataType
        from cosmos_predict2.models.utils import load_state_dict
        from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
        from imaginaire.constants import get_cosmos_predict2_video2world_checkpoint

        self._load_state_dict = load_state_dict
        self._video_dtype = DataType.VIDEO
        self.device = device
        self.torch_dtype = torch_dtype
        self.fps = fps

        config = get_cosmos_predict2_video2world_pipeline(
            model_size=model_size,
            resolution=resolution,
            fps=fps,
        )
        config.prompt_refiner_config.enabled = prompt_refiner_enabled
        config.guardrail_config.enabled = guardrail_enabled

        dit_path = get_cosmos_predict2_video2world_checkpoint(
            model_size=model_size,
            resolution=resolution,
            fps=fps,
        )
        self.pipe = Video2WorldPipeline.from_config(
            config=config,
            dit_path=dit_path,
            device=device,
            torch_dtype=torch_dtype,
        )

        if lora_checkpoint:
            resolved_checkpoint = self._resolve_lora_checkpoint(lora_checkpoint)
            self._inject_lora(
                checkpoint_path=resolved_checkpoint,
                rank=lora_rank,
                alpha=lora_alpha,
                target_modules=lora_target_modules,
            )

        self.latent_dim = int(self.pipe.tokenizer.latent_ch)
        self.hidden_dim = int(self.pipe.dit.model_channels)
        self.t_min = float(self.pipe.config.timestamps.t_min)
        self.t_max = float(self.pipe.config.timestamps.t_max)

    def _resolve_lora_checkpoint(self, checkpoint_path: str | Path) -> str:
        checkpoint_path = Path(checkpoint_path).expanduser()
        if checkpoint_path.is_file():
            return str(checkpoint_path.resolve())

        latest_file = checkpoint_path / "latest_checkpoint.txt"
        model_dir = checkpoint_path / "model"
        if latest_file.exists() and model_dir.exists():
            latest_name = latest_file.read_text().strip()
            candidate = model_dir / latest_name
            if candidate.exists():
                return str(candidate.resolve())

        raise FileNotFoundError(f"Unable to resolve Cosmos LoRA checkpoint from: {checkpoint_path}")

    def _inject_lora(self, checkpoint_path: str, rank: int, alpha: int, target_modules: str) -> None:
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights=True,
            target_modules=target_modules.split(","),
        )
        self.pipe.dit = inject_adapter_in_model(lora_cfg, self.pipe.dit)

        state_dict = self._load_state_dict(checkpoint_path)
        weights = {k[8:]: v for k, v in state_dict.items() if k.startswith("net_ema.")}
        if not weights:
            weights = {k[4:]: v for k, v in state_dict.items() if k.startswith("net.")}
        self.pipe.dit.load_state_dict(weights, strict=False)
        self.pipe.dit = self.pipe.dit.to(device=self.device, dtype=self.torch_dtype)

    def feature_num_frames(self, num_pixel_frames: int) -> int:
        return int(self.pipe.tokenizer.get_latent_num_frames(int(num_pixel_frames)))

    def _min_latent_frames_for_dit(self) -> int:
        """DiT uses 3D convs with temporal kernel 3; latents with T<3 raise in ``pipe.denoise``."""
        return 3

    @staticmethod
    def _to_bcthw_for_temporal_pad(frames: torch.Tensor) -> tuple[torch.Tensor, str]:
        """Normalize to ``B,C,T,H,W`` for time-axis padding; return layout tag to restore caller order.

        Callers differ: some pass ``B,C,T,H,W``, others ``B,T,C,H,W`` (e.g. LeRobot / test scripts).
        Padding the wrong axis turns RGB into a fake ``9``-channel tensor and breaks ``tokenizer.encode``.
        """
        expect_rank(frames, 5, "frames")
        b, d1, d2 = frames.shape[0], frames.shape[1], frames.shape[2]
        if d1 == 3 and d2 != 3:
            return frames, "bcthw"
        if d2 == 3 and d1 != 3:
            return frames.permute(0, 2, 1, 3, 4).contiguous(), "btchw"
        if d1 == 3 and d2 == 3:
            # [B,3,3,H,W] only: assume time-before-channels (matches common batch layout).
            return frames.permute(0, 2, 1, 3, 4).contiguous(), "btchw"
        raise ValueError(
            f"Cannot infer video layout for padding (expect RGB size 3 on dim 1 or 2): shape {tuple(frames.shape)}"
        )

    @staticmethod
    def _restore_frame_layout(bcthw: torch.Tensor, layout: str) -> torch.Tensor:
        if layout == "bcthw":
            return bcthw
        return bcthw.permute(0, 2, 1, 3, 4).contiguous()

    def _pad_pixel_frames_for_min_latent(self, frames: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Repeat the last frame so tokenizer latent length meets DiT minimum temporal extent."""
        bcthw, layout = self._to_bcthw_for_temporal_pad(frames)
        _b, _c, t_orig, _h, _w = bcthw.shape
        tok = self.pipe.tokenizer
        min_latent = self._min_latent_frames_for_dit()
        t_target = int(t_orig)
        while t_target < 2048 and int(tok.get_latent_num_frames(t_target)) < min_latent:
            t_target += 1
        if int(tok.get_latent_num_frames(t_target)) < min_latent:
            raise RuntimeError(
                f"Could not pad temporal length: pixel T={t_orig}..{t_target}, "
                f"need at least {min_latent} latent frames for DiT."
            )
        if t_target == t_orig:
            return self._restore_frame_layout(bcthw, layout), t_orig
        pad = t_target - t_orig
        last = bcthw[:, :, -1:, :, :].expand(-1, -1, pad, -1, -1)
        out = torch.cat([bcthw, last], dim=2)
        return self._restore_frame_layout(out, layout), t_orig

    @staticmethod
    def _slice_temporal_to_latent_tokens(
        tensor_3d: torch.Tensor,
        latent_time_full: int,
        latent_time_keep: int,
    ) -> torch.Tensor:
        """Pick the time axis matching ``latent_time_full`` and keep the first ``latent_time_keep`` steps."""
        if tensor_3d.dim() != 3:
            return tensor_3d
        for d in range(3):
            if int(tensor_3d.shape[d]) == latent_time_full:
                idx: list[slice | int] = [slice(None)] * 3
                idx[d] = slice(0, latent_time_keep)
                return tensor_3d[tuple(idx)].contiguous()
        return tensor_3d[:, :latent_time_keep, :].contiguous()

    def _prepare_video(self, frames: torch.Tensor) -> torch.Tensor:
        expect_rank(frames, 5, "frames")
        video = frames.to(device=self.device)
        if video.dtype == torch.uint8:
            video = video.to(torch.float32) / 127.5 - 1.0
        else:
            video = video.to(self.torch_dtype)
            if float(video.min().item()) >= -1.01 and float(video.max().item()) <= 1.01:
                if float(video.min().item()) >= -0.01:
                    video = video * 2.0 - 1.0
            else:
                video = video / 127.5 - 1.0
        video = video.clamp(-1.0, 1.0)
        return video.permute(0, 2, 1, 3, 4).contiguous()

    def _run_tokenizer_encode(self, frames: torch.Tensor) -> _PolicyCosmosEncodeOutput:
        """VAE encode only (no temporal padding). Used after padding or when full latent length is needed for DiT."""
        video = self._prepare_video(frames)
        latent = self.pipe.encode(video)
        tokens = latent.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        return _PolicyCosmosEncodeOutput(latent=latent, tokens=tokens)

    def encode_tokens(self, frames: torch.Tensor) -> _PolicyCosmosEncodeOutput:
        """Encode pixels to latent tokens; pads time so tokenizer 3D convs see enough frames, then trims to logical length."""
        frames_in, t_pixel_orig = self._pad_pixel_frames_for_min_latent(frames)
        latent_keep = self.feature_num_frames(t_pixel_orig)
        out = self._run_tokenizer_encode(frames_in)
        t_full = int(out.latent.shape[2])
        keep = min(latent_keep, t_full)
        latent = out.latent[:, :, :keep, :, :]
        tokens = latent.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        return _PolicyCosmosEncodeOutput(latent=latent, tokens=tokens)

    def _build_condition(self, latent: torch.Tensor, texts: list[str] | None) -> Any:
        batch_size, _, _, height, width = latent.shape
        prompts = texts if texts is not None and len(texts) == batch_size else [""] * batch_size
        prompt_embeddings = self.pipe.encode_prompt(prompts).to(device=self.device, dtype=self.torch_dtype)
        data_batch = {
            "t5_text_embeddings": prompt_embeddings,
            "fps": torch.full((batch_size,), self.fps, device=self.device, dtype=torch.long),
            "padding_mask": torch.zeros(batch_size, 1, height, width, device=self.device, dtype=self.torch_dtype),
        }
        condition = self.pipe.conditioner(data_batch)
        condition = condition.edit_data_type(self._video_dtype)
        return condition.set_video_condition(
            gt_frames=latent.to(device=self.device, dtype=self.torch_dtype),
            random_min_num_conditional_frames=0,
            random_max_num_conditional_frames=0,
            num_conditional_frames=0,
        )

    def _sigma_from_tau(self, tau: float, noise_level: float) -> float:
        if tau <= 1.0:
            base_sigma = math.exp(math.log(self.t_min) + tau * (math.log(self.t_max) - math.log(self.t_min)))
        else:
            base_sigma = tau
        scaled_sigma = base_sigma * max(noise_level, 0.0)
        return float(min(max(scaled_sigma, self.t_min), self.t_max))

    def _capture_hidden_tokens(
        self,
        xt_latent: torch.Tensor,
        sigma: float,
        condition: Any,
    ) -> list[torch.Tensor]:
        captured: list[torch.Tensor] = []
        hooks: list[Any] = []

        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Unexpected Cosmos block output type: {type(output)}")
            pooled = output.mean(dim=(2, 3)).float().contiguous()
            captured.append(pooled)

        for block in self.pipe.dit.blocks:
            hooks.append(block.register_forward_hook(_hook))

        sigma_B_T = torch.full(
            (xt_latent.shape[0], xt_latent.shape[2]),
            fill_value=sigma,
            device=xt_latent.device,
            dtype=xt_latent.dtype,
        )

        try:
            with torch.no_grad():
                self.pipe.denoise(xt_latent, sigma_B_T, condition)
        finally:
            for hook in hooks:
                hook.remove()

        return captured

    def denoise_to_tau(
        self,
        frames: torch.Tensor,
        tau: float,
        noise_level: float,
        texts: list[str] | None = None,
    ) -> _PolicyCosmosRuntimeDenoiseOutput:
        frames_in, t_pixel_orig = self._pad_pixel_frames_for_min_latent(frames)
        latent_time_keep = self.feature_num_frames(t_pixel_orig)

        encoded = self._run_tokenizer_encode(frames_in)
        latent_time_full = int(encoded.latent.shape[2])
        sigma = self._sigma_from_tau(tau=tau, noise_level=noise_level)
        condition = self._build_condition(encoded.latent, texts)

        noise = torch.randn_like(encoded.latent) * sigma
        xt_latent = encoded.latent + noise
        hidden_tokens = self._capture_hidden_tokens(xt_latent=xt_latent, sigma=sigma, condition=condition)

        enc_tok = encoded.tokens.float()
        enc_tok = self._slice_temporal_to_latent_tokens(enc_tok, latent_time_full, latent_time_keep)
        hidden_tokens = [
            self._slice_temporal_to_latent_tokens(h.float(), latent_time_full, latent_time_keep)
            for h in hidden_tokens
        ]

        return _PolicyCosmosRuntimeDenoiseOutput(
            encoded_tokens=enc_tok,
            hidden_tokens=hidden_tokens,
            sigma=sigma,
        )


class CosmosAdapter(nn.Module):
    """Policy-facing Cosmos boundary: lightweight encoder fallback + optional Predict2 runtime."""

    def __init__(
        self,
        feature_dim: int,
        hidden_layer_index: int = -1,
        noise_level: float = 0.0,
        num_hidden_layers: int = 8,
        image_channels: int = 3,
        device: str = "cuda",
        use_external_runtime: bool = False,
        external_module: str | None = None,
        repo_path: str | None = None,
        model_size: str = "2B",
        resolution: str = "480",
        fps: int = 10,
        lora_checkpoint: str | None = None,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
        prompt_refiner_enabled: bool = False,
        guardrail_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_layer_index = hidden_layer_index
        self.noise_level = noise_level
        self.num_hidden_layers = num_hidden_layers
        self.device_spec = device
        self.use_external_runtime = use_external_runtime
        self.external_module = external_module
        self.repo_path = repo_path
        self.model_size = model_size
        self.resolution = resolution
        self.fps = fps
        self.lora_checkpoint = lora_checkpoint
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.prompt_refiner_enabled = prompt_refiner_enabled
        self.guardrail_enabled = guardrail_enabled

        self.runtime: Any | None = None
        self.runtime_error: Exception | None = None
        self.runtime_encode_proj: nn.Module = nn.Identity()
        self.runtime_hidden_proj: nn.Module = nn.Identity()
        self._runtime_encode_in_dim: int | None = None
        self._runtime_hidden_in_dim: int | None = None

        self.encoder = SimpleVisualFeatureExtractor(image_channels, feature_dim)
        self.hidden_stack = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, feature_dim * 4),
                    nn.GELU(approximate="tanh"),
                    nn.Linear(feature_dim * 4, feature_dim),
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def _default_repo_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "cosmos-predict2"

    def _ensure_runtime_projections(self, encode_dim: int, hidden_dim: int, device: torch.device) -> None:
        if self._runtime_encode_in_dim != encode_dim:
            self.runtime_encode_proj = nn.Identity() if encode_dim == self.feature_dim else nn.Linear(encode_dim, self.feature_dim)
            self.runtime_encode_proj.to(device=device)
            self._runtime_encode_in_dim = encode_dim
        if self._runtime_hidden_in_dim != hidden_dim:
            self.runtime_hidden_proj = nn.Identity() if hidden_dim == self.feature_dim else nn.Linear(hidden_dim, self.feature_dim)
            self.runtime_hidden_proj.to(device=device)
            self._runtime_hidden_in_dim = hidden_dim

    def _ensure_runtime(self) -> None:
        if self.runtime is not None or self.runtime_error is not None or not self.use_external_runtime:
            return

        try:
            if self.external_module:
                self.runtime = importlib.import_module(self.external_module)
                return

            # None / "" = rely on installed cosmos_predict2; do not force horizon's sibling clone path.
            runtime_repo_path = self.repo_path if self.repo_path else None
            self.runtime = _CosmosPredict2Runtime(
                repo_path=runtime_repo_path,
                device=self.device_spec,
                torch_dtype=torch.bfloat16,
                model_size=self.model_size,
                resolution=self.resolution,
                fps=self.fps,
                lora_checkpoint=self.lora_checkpoint,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                prompt_refiner_enabled=self.prompt_refiner_enabled,
                guardrail_enabled=self.guardrail_enabled,
            )
        except Exception as exc:  # pragma: no cover - exercised only when runtime deps are present/missing.
            self.runtime_error = exc
            warnings.warn(
                f"Failed to initialize Cosmos Predict2 runtime: {exc}. Falling back to the local lightweight encoder.",
                stacklevel=2,
            )

    def feature_num_frames(self, num_input_frames: int) -> int:
        self._ensure_runtime()
        if self.runtime is not None and hasattr(self.runtime, "feature_num_frames"):
            return int(self.runtime.feature_num_frames(num_input_frames))
        return int(num_input_frames)

    def _encode_fallback(self, frames: torch.Tensor) -> torch.Tensor:
        expect_rank(frames, 5, "frames")
        return self.encoder(frames)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self._ensure_runtime()
        if self.runtime is not None and hasattr(self.runtime, "encode_tokens"):
            encoded = self.runtime.encode_tokens(frames)
            self._ensure_runtime_projections(
                encode_dim=int(encoded.tokens.shape[-1]),
                hidden_dim=getattr(self.runtime, "hidden_dim", int(encoded.tokens.shape[-1])),
                device=encoded.tokens.device,
            )
            return self.runtime_encode_proj(encoded.tokens.float())
        return self._encode_fallback(frames)

    def encode_multiview(self, frame_views: list[torch.Tensor]) -> torch.Tensor:
        encoded_views = [self.encode(view) for view in frame_views]
        return torch.stack(encoded_views, dim=0).mean(dim=0)

    def _denoise_fallback(
        self,
        frames_or_features: torch.Tensor,
        tau: float,
        noise_level: float | None,
    ) -> CosmosDenoiseState:
        if frames_or_features.ndim == 5:
            encoded = self._encode_fallback(frames_or_features)
        elif frames_or_features.ndim == 3:
            encoded = frames_or_features
        else:
            raise ValueError(f"Expected rank 3 or 5 input, got shape {tuple(frames_or_features.shape)}")

        resolved_noise = self.noise_level if noise_level is None else noise_level
        x = encoded
        if resolved_noise > 0.0 and tau > 0.0:
            x = x + torch.randn_like(x) * resolved_noise * tau

        hidden_states: list[torch.Tensor] = []
        for block in self.hidden_stack:
            x = x + block(x)
            hidden_states.append(x)

        return CosmosDenoiseState(
            encoded_features=encoded,
            hidden_states=hidden_states,
            tau=float(tau),
            noise_level=float(resolved_noise),
            sigma=float(resolved_noise * tau),
        )

    def denoise_to_tau(
        self,
        frames_or_features: torch.Tensor,
        tau: float,
        hidden_layer_index: int | None = None,
        noise_level: float | None = None,
        texts: list[str] | None = None,
    ) -> CosmosDenoiseState:
        del hidden_layer_index
        self._ensure_runtime()
        if self.runtime is not None and hasattr(self.runtime, "denoise_to_tau") and self.external_module:
            return self.runtime.denoise_to_tau(
                frames_or_features,
                tau=tau,
                hidden_layer_index=hidden_layer_index,
                noise_level=noise_level,
            )

        if self.runtime is not None and hasattr(self.runtime, "denoise_to_tau") and frames_or_features.ndim == 5:
            resolved_noise = self.noise_level if noise_level is None else noise_level
            denoise_output = self.runtime.denoise_to_tau(
                frames=frames_or_features,
                tau=tau,
                noise_level=resolved_noise,
                texts=texts,
            )
            hidden_dim = int(denoise_output.hidden_tokens[0].shape[-1]) if denoise_output.hidden_tokens else self.feature_dim
            self._ensure_runtime_projections(
                encode_dim=int(denoise_output.encoded_tokens.shape[-1]),
                hidden_dim=hidden_dim,
                device=denoise_output.encoded_tokens.device,
            )
            encoded_features = self.runtime_encode_proj(denoise_output.encoded_tokens.float())
            hidden_states = [self.runtime_hidden_proj(hidden.float()) for hidden in denoise_output.hidden_tokens]
            return CosmosDenoiseState(
                encoded_features=encoded_features,
                hidden_states=hidden_states,
                tau=float(tau),
                noise_level=float(resolved_noise),
                sigma=float(denoise_output.sigma),
            )

        return self._denoise_fallback(frames_or_features=frames_or_features, tau=tau, noise_level=noise_level)

    def get_nth_hidden_layer(
        self,
        denoise_state: CosmosDenoiseState,
        hidden_layer_index: int | None = None,
    ) -> torch.Tensor:
        if self.runtime is not None and hasattr(self.runtime, "get_nth_hidden_layer"):
            return self.runtime.get_nth_hidden_layer(denoise_state, hidden_layer_index=hidden_layer_index)

        if not denoise_state.hidden_states:
            raise ValueError("No hidden states are available.")
        index = self.hidden_layer_index if hidden_layer_index is None else hidden_layer_index
        return denoise_state.hidden_states[index]

    def extract_hidden_features(
        self,
        frames_or_features: torch.Tensor,
        tau: float,
        hidden_layer_index: int | None = None,
        noise_level: float | None = None,
        texts: list[str] | None = None,
    ) -> torch.Tensor:
        denoise_state = self.denoise_to_tau(
            frames_or_features=frames_or_features,
            tau=tau,
            hidden_layer_index=hidden_layer_index,
            noise_level=noise_level,
            texts=texts,
        )
        return self.get_nth_hidden_layer(denoise_state, hidden_layer_index=hidden_layer_index)

    def extract_hidden_features_multiview(
        self,
        frame_views: list[torch.Tensor],
        tau: float,
        hidden_layer_index: int | None = None,
        noise_level: float | None = None,
        texts: list[str] | None = None,
    ) -> torch.Tensor:
        per_view = [
            self.extract_hidden_features(
                frames_or_features=view,
                tau=tau,
                hidden_layer_index=hidden_layer_index,
                noise_level=noise_level,
                texts=texts,
            )
            for view in frame_views
        ]
        return torch.stack(per_view, dim=0).mean(dim=0)

    def empty_future_features(
        self,
        batch_size: int,
        future_window: int,
        device: torch.device,
    ) -> torch.Tensor:
        feature_window = self.feature_num_frames(future_window)
        return torch.zeros(batch_size, feature_window, self.feature_dim, device=device)
