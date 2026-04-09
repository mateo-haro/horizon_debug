from __future__ import annotations

import importlib
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lerobot_policy_horizon.backbones.visual_encoder import SimpleVisualEncoder
from lerobot_policy_horizon.utils.shapes import expect_rank


@dataclass
class CosmosDenoiseState:
    encoded_features: torch.Tensor
    hidden_states: list[torch.Tensor]
    tau: float
    noise_level: float
    sigma: float | None = None


@dataclass
class _CosmosEncodeOutput:
    latent: torch.Tensor
    tokens: torch.Tensor


@dataclass
class _CosmosRuntimeDenoiseOutput:
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
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise FileNotFoundError(f"Cosmos repo path does not exist: {repo_path}")
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise RuntimeError("Cosmos Predict2 runtime requires CUDA for practical use.")

        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

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

    def encode_tokens(self, frames: torch.Tensor) -> _CosmosEncodeOutput:
        video = self._prepare_video(frames)
        latent = self.pipe.encode(video)
        tokens = latent.mean(dim=(-1, -2)).transpose(1, 2).contiguous()
        return _CosmosEncodeOutput(latent=latent, tokens=tokens)

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
        # A zero-frame mask yields a valid "no video condition" denoising path while still matching
        # the expected DiT input contract of the LIBERO fine-tuned checkpoint.
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
        hooks = []

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
    ) -> _CosmosRuntimeDenoiseOutput:
        encoded = self.encode_tokens(frames)
        sigma = self._sigma_from_tau(tau=tau, noise_level=noise_level)
        condition = self._build_condition(encoded.latent, texts)

        noise = torch.randn_like(encoded.latent) * sigma
        xt_latent = encoded.latent + noise
        hidden_tokens = self._capture_hidden_tokens(xt_latent=xt_latent, sigma=sigma, condition=condition)

        return _CosmosRuntimeDenoiseOutput(
            encoded_tokens=encoded.tokens.float(),
            hidden_tokens=hidden_tokens,
            sigma=sigma,
        )


class CosmosAdapter(nn.Module):
    """Repository-local boundary around the LIBERO-tuned Cosmos Predict2 fork."""

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

        self.encoder = SimpleVisualEncoder(image_channels, feature_dim)
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

            runtime_repo_path = self.repo_path or str(self._default_repo_path())
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
