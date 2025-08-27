# COBOT Finetune Debug Guide

This guide helps debug NaN issues when running finetuning on COBOT dataset.

## Common Causes of NaN Errors

1. **Learning Rate Too High**: Finetuning requires much lower learning rates
2. **Missing Gradient Clipping**: Large gradients can cause instability
3. **Batch Size Too Large**: Can cause memory and numerical issues
4. **Data Normalization Issues**: Check if your data is properly normalized

## Solutions Applied

### 1. Reduced Learning Rate
```yaml
# Original (causing NaN)
base_lr: 0.1

# Fixed (stable)
base_lr: 0.01
```

### 2. Added Gradient Clipping
```yaml
# Added to config
grad_clip: 1.0
```

### 3. Reduced Batch Size
```yaml
# Original
batch_size: 64

# Fixed
batch_size: 32
```

### 4. Added Weight Decay
```yaml
# Original
weight_decay: 0.0

# Fixed
weight_decay: 0.0001
```

## Running COBOT Finetuning

### Method 1: Use the Scripts
```bash
# Windows
run_cobot_finetune.bat

# Linux/Mac
chmod +x run_cobot_finetune.sh
./run_cobot_finetune.sh
```

### Method 2: Manual Command
```bash
python main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub.yaml
```

## Configuration File

The COBOT finetune config is at:
`config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub.yaml`

Key settings:
```yaml
# Stable finetuning settings
base_lr: 0.01
batch_size: 32
weight_decay: 0.0001
grad_clip: 1.0
unfreeze_backbone: True
```

## Troubleshooting Steps

### 1. Check Pretrained Model
Make sure your pretrained model exists:
```bash
ls -la ./work_dir/cobot_3views_2D_xsub_medgap/pretext/epoch400_model.pt
```

### 2. Check Data
Verify your data files exist:
```bash
ls -la ./cobot_med_frame64/xsub/train_position.npy
ls -la ./cobot_med_frame64/xsub/train_label.pkl
```

### 3. Monitor Training
Watch for these signs:
- Loss should decrease, not become NaN
- Learning rate should be 0.01 initially
- Gradients should be clipped if they exceed 1.0

### 4. If Still Getting NaN

Try even more conservative settings:
```yaml
base_lr: 0.005  # Even lower
batch_size: 16   # Smaller batch
grad_clip: 0.5   # Tighter clipping
```

### 5. Check Data Quality
```python
import numpy as np

# Load your data and check for issues
data = np.load('./cobot_med_frame64/xsub/train_position.npy')
print(f"Data shape: {data.shape}")
print(f"Data range: {data.min():.3f} to {data.max():.3f}")
print(f"Data mean: {data.mean():.3f}")
print(f"Data std: {data.std():.3f}")
print(f"NaN count: {np.isnan(data).sum()}")
print(f"Inf count: {np.isinf(data).sum()}")
```

## Expected Behavior

With the fixed configuration:
1. **Epoch 1**: Loss should be around 2.5-3.0, not NaN
2. **Learning Rate**: Starts at 0.01, decreases to 0.001 at epoch 80
3. **Convergence**: Loss should steadily decrease
4. **Final Accuracy**: Should be higher than linear evaluation

## Output Location

Results will be saved in:
`./work_dir/cobot_3views_2D_xsub_medgap/finetune/`

## Notes

- The gradient clipping feature was added to the linear evaluation processor
- COBOT dataset may require different hyperparameters than NTU60
- Monitor GPU memory usage during training
- If issues persist, try running linear evaluation first to verify data loading
