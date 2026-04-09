# Horizon

`Horizon` mimic-video inspired Video Action Model.

The core idea is unchanged:

- action policy: robotics-specific ActionDiT trained with flow matching
- conditioning: current visual features, future visual features, optional proprio, optional task text
- world-model boundary: Cosmos stays external behind adapters
- future signal: clean encodings, noised encodings, or intermediate denoiser hidden layers at noise level `tau`

Dependencies:

- datasets: reuse public LeRobot LIBERO datasets on the Hugging Face Hub
- training loop: target LeRobot's train loop through a custom policy plugin package
- task metadata: reuse public LIBERO suite classes 

This repo owns only the parts that are actually custom:

- `horizon_dit` policy config and model
- ActionDiT implementation
- future-source abstraction
- Cosmos encoding / hidden-layer adapter
- thin training launcher and AWS bootstrap

## Architecture

The policy package lives in [src/lerobot_policy_horizon](/media/eandre/PortableSSD/ETHRC/horizon/src/lerobot_policy_horizon).

The important boundary is the Cosmos adapter:

- `encode(...)`
- `denoise_to_tau(...)`
- `get_nth_hidden_layer(...)`
- `extract_hidden_features(...)`

This lets you do exactly:

1. encode current or future frames
2. denoise to a chosen `tau`
3. extract hidden layer `n`

The policy then conditions ActionDiT on:

```python
{
    "curr_vis": Tensor[B, T_curr, D_vis],
    "future_vis": Tensor[B, T_future, D_vis],
    "future_mask": BoolTensor[B, T_future],
    "proprio": Optional[Tensor[B, T_curr, D_prop]],
    "text": list[str],
    "future_source_id": LongTensor[B],
}
```

Supported future sources:

- `current_only`
- `oracle_clean`
- `oracle_noised`
- `oracle_hidden`
- `mixed`
- `generated` For actual inference

## Repo Tree

```text
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/train/
│   └── libero_horizon_oracle_hidden.yaml
├── scripts/
│   ├── aws_bootstrap.sh
│   ├── test_horizon_stack.py
│   └── train_horizon_libero.py
├── src/
│   ├── horizon/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── lerobot.py
│   │   └── libero.py
│   └── lerobot_policy_horizon/
│       ├── __init__.py
│       ├── compat.py
│       ├── configuration_horizon_dit.py
│       ├── modeling_horizon_dit.py
│       ├── processor_horizon_dit.py
│       ├── backbones/
│       ├── future_sources/
│       ├── models/
│       └── utils/
└── tests/
```

## Public Components Reused

Datasets:

- `lerobot/libero_spatial_image`
- `lerobot/libero_object_image`
- `lerobot/libero_goal_image`
- `lerobot/libero_10_image`
- `lerobot/libero_90_image`

Training loop:

- `lerobot-train`

Task metadata / suite definitions:

- official LIBERO benchmark classes when `libero` is installed

## Quickstart

Create the environment on an AWS box:

```bash
bash scripts/aws_bootstrap.sh
source .venv/bin/activate
```

Print the exact LeRobot launch command:

```bash
python3 scripts/train_horizon_libero.py --dry-run
```

Launch training:

```bash
python3 scripts/train_horizon_libero.py
```

Pass additional LeRobot overrides after `--`:

```bash
python3 scripts/train_horizon_libero.py -- \
  --steps=200000 \
  --wandb.enable=true
```

## Interactive AWS Test

For an interactive GPU session, the practical sequence is:

1. get a shell inside the AWS pod
2. clone or copy `horizon`
3. install `horizon` editable
4. install the full Cosmos fork environment from `cosmos-predict2/scripts/libero.md`
5. run the diagnostic script before attempting training

Inside the pod:

```bash
cd /workspace
git clone <your-horizon-repo-url> horizon
cd horizon
python -m pip install -e .
python -m pip install "lerobot[libero]"
```

For real Cosmos runtime support, `aws_bootstrap.sh` is not enough. You also need the Cosmos fork dependencies:

```bash
cd /workspace/horizon/cosmos-predict2
python -m pip install uv
uv sync --extra cu126
source .venv/bin/activate
cd /workspace/horizon
python -m pip install -e .
python -m pip install "lerobot[libero]"
```

Download the base Cosmos checkpoint:

```bash
cd /workspace/horizon/cosmos-predict2
python scripts/download_checkpoints.py --model_types video2world --model_sizes 2B --resolution 480 --fps 10
```

If you also want the LIBERO LoRA weights, point the diagnostic and config at the checkpoint directory or file later with `--cosmos-lora-checkpoint`.

Run the diagnostic:

```bash
cd /workspace/horizon
python scripts/test_horizon_stack.py
```

If you only want to verify the Horizon side first and skip real Cosmos initialization:

```bash
python scripts/test_horizon_stack.py --skip-real-cosmos
```

## Test Output

The diagnostic script prints `PASS`, `WARN`, and `FAIL` lines.

Typical early bring-up output:

- `PASS repo_root`, `PASS cosmos_repo`, `PASS system`: repo and GPU are visible
- `PASS launcher_dry_run`: the LeRobot launch command is well-formed
- `PASS fallback_adapter`: the local non-Cosmos path works
- `PASS policy_forward`: the Horizon policy can run a synthetic forward pass
- `WARN import:omegaconf`, `WARN import:hydra`, `WARN import:peft`, etc.: Cosmos dependencies are still missing
- `WARN real_cosmos_adapter: runtime did not initialize ...`: expected until the Cosmos fork env is installed correctly

What you want before trying real training:

- `PASS import:cosmos_predict2`
- no critical Cosmos dependency warnings
- `PASS real_cosmos_encode`
- `PASS real_cosmos_denoise`
- `PASS policy_forward`

If the script ends with `FAIL`, do not start training yet. Fix the failing dependency or runtime issue first.

## Config

Default config:

- [libero_horizon_oracle_hidden.yaml](/media/eandre/PortableSSD/ETHRC/horizon/configs/train/libero_horizon_oracle_hidden.yaml)

Important fields:

- `train.suite`: LIBERO suite mapped to a public LeRobot dataset repo
- `policy.future_source`: `oracle_clean`, `oracle_noised`, `oracle_hidden`, `mixed`, ...
- `policy.cosmos_hidden_layer`
- `policy.cosmos_noise_level`
- `policy.cosmos_tau`
- `policy.cosmos_repo_path`
- `policy.cosmos_model_size`
- `policy.cosmos_resolution`
- `policy.cosmos_fps`
- `policy.cosmos_lora_checkpoint`
- `policy.current_obs_steps`
- `policy.future_obs_steps`
- `policy.future_offset`

The observation delta indices are derived automatically from:

- current observation steps
- future observation steps
- future offset

So the LeRobot dataset loader can provide both current and oracle-future frames without a custom dataset implementation.

## Cosmos Adapter

The adapter in [cosmos_adapter.py](/media/eandre/PortableSSD/ETHRC/horizon/src/lerobot_policy_horizon/backbones/cosmos_adapter.py) now has two paths:

- real runtime path for the local `cosmos-predict2` `libero` branch
- lightweight fallback path so the repo still shape-tests without Cosmos dependencies

The real path follows your fork’s LIBERO notes in `cosmos-predict2/scripts/libero.md`:

- model family: `Cosmos-Predict2-2B-Video2World`
- checkpoint variant: `480p`, `10fps`
- optional LoRA injection from the LIBERO fine-tune checkpoint

The adapter exposes:

- `encode(...)`: tokenizer latent features, pooled over space and projected to Horizon feature dim
- `denoise_to_tau(...)`: noise latent features to a sigma derived from `tau`, run the real Cosmos DiT, and capture per-block hidden states
- `get_nth_hidden_layer(...)`: select the requested hidden layer after projection to Horizon feature dim

If the Cosmos runtime cannot be imported, Horizon falls back to the local lightweight encoder and emits a warning instead of breaking import-time tests.

## Notes

- This repo does not vendor Cosmos.
- This repo does not vendor LIBERO datasets.
- This repo does not replace the LeRobot train loop.
- `generated` future conditioning is still a TODO because it requires a real frozen Cosmos rollout path.
