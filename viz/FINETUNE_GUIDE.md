# Finetune Evaluation Guide for AimCLR-v2

This guide shows how to run finetune evaluation using the existing `linear_evaluation` processor.

## Overview

The finetune evaluation protocol tests performance when finetuning the entire network with all labeled data. This is already implemented in the existing `linear_evaluation.py` processor through the `unfreeze_backbone` parameter.

## Key Differences from Linear Evaluation

| Aspect | Linear Evaluation | Finetune Evaluation |
|--------|-------------------|---------------------|
| **Layers Trained** | Only FC layers | All layers |
| **Learning Rate** | 3.0 → 0.3 | 0.1 → 0.01 |
| **Parameter** | `unfreeze_backbone: False` | `unfreeze_backbone: True` |
| **Purpose** | Test representation quality | Test full network performance |

## Running Finetune Evaluation

### Prerequisites

1. **Pretrained Model**: You need a pretrained model from the pretext task
2. **Data**: NTU60 dataset should be prepared
3. **Updated Config**: Use the finetune configuration files

### Method 1: Using the Scripts

#### Linux/Mac
```bash
chmod +x run_finetune.sh
./run_finetune.sh
```

#### Windows
```bash
run_finetune.bat
```

### Method 2: Manual Commands

```bash
# NTU60 xsub
python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xsub.yaml

# NTU60 xview
python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xview.yaml
```

### Method 3: Modify Existing Config

You can also modify any existing linear evaluation config by adding:

```yaml
# Add this line to any linear evaluation config
unfreeze_backbone: True

# And change the learning rate
base_lr: 0.1  # instead of 3.0
step: [80]    # instead of [80] (same but different reduction)
```

## Configuration Files

The finetune configs are located at:
- `config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xsub.yaml`
- `config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xview.yaml`

### Key Settings in Finetune Configs

```yaml
# Finetune: unfreeze all layers
unfreeze_backbone: True

# Lower learning rate for finetuning
base_lr: 0.1

# Learning rate schedule (0.1 → 0.01 at epoch 80)
step: [80]

# All other settings remain the same as linear evaluation
```

## Expected Results

Finetune evaluation typically achieves higher accuracy than linear evaluation because it can adapt all layers to the specific task:

| Protocol | NTU 60 xsub (%) | NTU 60 xview (%) |
|----------|-----------------|------------------|
| Linear Evaluation | ~80.9 | ~85.4 |
| **Finetune Evaluation** | **~85-90** | **~90-95** |

## Output Locations

Results are saved in:
- NTU60 xsub: `./work_dir/ntu60_cs/finetune/`
- NTU60 xview: `./work_dir/ntu60_cv/finetune/`

## Troubleshooting

1. **Pretrained Model Missing**: Update the `weights` path in config files to point to your pretrained model
2. **Memory Issues**: Reduce batch size if needed
3. **Learning Rate**: If training is unstable, try reducing `base_lr` to 0.05 or 0.01

## Notes

- Finetuning takes longer than linear evaluation because all layers are being updated
- The learning rate is lower to prevent catastrophic forgetting
- Results may vary depending on the quality of the pretrained model
