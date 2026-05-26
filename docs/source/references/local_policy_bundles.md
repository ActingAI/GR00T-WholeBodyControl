# Local SONIC Policy Bundles

This fork keeps exported SONIC deployment bundles out of git. ONNX, TensorRT
engines, exported checkpoints, and transfer archives are large machine-local
artifacts. They should live under:

```text
gear_sonic_deploy/policy/local/
```

That directory is ignored by git. The code contains paths and documentation for
the expected layout, but the model files themselves are not committed.

## Expected Layout

```text
gear_sonic_deploy/policy/local/
  base_public_pt_041550/
    model_encoder.onnx
    model_decoder.onnx
    observation_config.yaml
    manifest.json
  base_public_pt_release_schema_041550/
    model_encoder.onnx
    model_decoder.onnx
    observation_config.yaml
    manifest.json
    onnx_check_summary.json
  finetuned/
    conservative_A_000500/
    aggressive_B_002000/
    optimal_B_003500/
  finetuned_release_schema/
    conservative_A_000500/
    aggressive_B_002000/
    optimal_B_003500/
```

The files required by `deploy.sh --cp <prefix>` are:

```text
<prefix>_encoder.onnx
<prefix>_decoder.onnx
observation_config.yaml
```

For example, the deploy prefix:

```text
policy/local/finetuned_release_schema/optimal_B_003500/model
```

expects:

```text
gear_sonic_deploy/policy/local/finetuned_release_schema/optimal_B_003500/model_encoder.onnx
gear_sonic_deploy/policy/local/finetuned_release_schema/optimal_B_003500/model_decoder.onnx
gear_sonic_deploy/policy/local/finetuned_release_schema/optimal_B_003500/observation_config.yaml
```

TensorRT `.trt` files are generated on first deploy and should remain local.

## Release-Schema Bundles

Use the `*_release_schema` bundles for normal real-robot or sim teleop testing.
They match the official release deployment interface:

```text
encoder input: [1, 1762]
decoder input: [1, 994]
```

The older direct exports under `finetuned/` and `base_public_pt_041550/` use a
1751-D encoder input. They are kept only for diagnostics and comparison.

The 11-D difference is:

```text
motion_root_z_position_10frame_step5  # 10 dims
motion_root_z_position                # 1 dim
```

The release-schema bundles keep those slots so the C++ observation builder and
ONNX encoder agree on `[1, 1762]`.

## Launcher Presets

The data collection launcher has policy presets that point to the local
release-schema bundles:

| Preset | Deploy prefix |
| --- | --- |
| `release` | upstream `policy/release` default |
| `base` | `policy/local/base_public_pt_release_schema_041550/model` |
| `conservative` | `policy/local/finetuned_release_schema/conservative_A_000500/model` |
| `aggressive` | `policy/local/finetuned_release_schema/aggressive_B_002000/model` |
| `optimal` | `policy/local/finetuned_release_schema/optimal_B_003500/model` |

Example:

```bash
cd ~/GR00T-WholeBodyControl
python gear_sonic/scripts/launch_data_collection.py \
  --deploy-policy optimal \
  --camera-host 192.168.123.164 \
  --task-prompt "blanket_yellow_test"
```

Direct deploy is also supported:

```bash
cd ~/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh \
  --cp policy/local/finetuned_release_schema/optimal_B_003500/model \
  --obs-config policy/local/finetuned_release_schema/optimal_B_003500/observation_config.yaml \
  --input-type zmq_manager \
  real
```

## Verification

Use the included verification tools before testing a local bundle:

```bash
cd ~/GR00T-WholeBodyControl
python tools/verify_sonic_deploy_bundle.py \
  gear_sonic_deploy/policy/local/finetuned_release_schema/optimal_B_003500
```

Expected output:

```text
model_encoder.onnx:
  inputs:  [('obs_dict', [1, 1762])]
  outputs: [('encoded_tokens', [1, 64])]
model_decoder.onnx:
  inputs:  [('obs_dict', [1, 994])]
  outputs: [('action', [1, 29])]
OK: bundle is release-schema compatible.
```

For the base public `.pt` release-schema bundle:

```bash
python tools/verify_base_release_schema.py \
  gear_sonic_deploy/policy/local/base_public_pt_release_schema_041550
```

## What Not To Commit

Do not commit:

- `.onnx`
- `.trt`
- `.pt`, `.pth`, `.ckpt`, `.safetensors`
- zip/tar transfer archives
- generated `outputs/`

If a bundle needs to be transferred, copy or download it into
`gear_sonic_deploy/policy/local/` after cloning the repository.
