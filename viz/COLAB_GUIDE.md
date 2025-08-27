# AimCLR Training on Google Colab - Complete Guide

This guide will help you train AimCLR on your COBOT dataset using Google Colab's free GPU resources.

## 🚀 Quick Start (Recommended)

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. **Enable GPU**: Runtime → Change runtime type → Hardware accelerator → GPU

### Step 2: Upload Files
1. **Upload the setup script**: Upload `colab_setup.py` to Colab
2. **Upload your code files**: Upload all your AimCLR Python files
3. **Upload your dataset**: Upload your `cleaned_actions` folder as a ZIP file

### Step 3: Run Training
```python
# Run the automated setup
!python colab_setup.py
```

That's it! The script will handle everything automatically.

---

## 📋 Detailed Step-by-Step Guide

### Step 1: Prepare Your Files

**Before starting Colab, prepare these files:**

1. **Your AimCLR code files:**
   - `main.py`
   - `cleaned_to_ntu.py`
   - `net/aimclr_v2_3views.py`
   - `processor/linear_evaluation.py`
   - `feeder/ntu_feeder.py`
   - `feeder/tools.py`
   - `net/st_gcn.py`
   - `net/utils/graph.py`
   - `processor/processor.py`
   - `processor/pretrain_aimclr_v2_3views.py`

2. **Your dataset:**
   - Zip your `cleaned_actions` folder
   - Structure should be: `cleaned_actions/A1/`, `cleaned_actions/A2/`, etc.

### Step 2: Colab Setup

**Cell 1: Check GPU**
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**Cell 2: Install Dependencies**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install numpy scipy matplotlib tqdm pyyaml tensorboard opencv-python scikit-learn
```

**Cell 3: Upload Files**
```python
from google.colab import files

# Upload your code files
print("Upload your AimCLR Python files:")
uploaded = files.upload()

# Upload your dataset
print("Upload your cleaned_actions.zip:")
dataset = files.upload()
```

**Cell 4: Extract Dataset**
```python
import zipfile
for filename in dataset.keys():
    if filename.endswith('.zip'):
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('.')
        print(f"Extracted: {filename}")
```

**Cell 5: Run Training**
```python
# Run the automated setup script
!python colab_setup.py
```

---

## ⚙️ Manual Configuration (Alternative)

If you prefer manual control, here's how to set up everything step by step:

### Step 1: Create Directory Structure
```python
!mkdir -p config/cobot/pretext config/cobot/linear
!mkdir -p work_dir net processor feeder
```

### Step 2: Create Configuration Files

**Pretraining Config:**
```python
pretext_config = """
work_dir: ./work_dir/cobot_3views_2D_cleaned/pretext

# feeder
train_feeder: feeder.ntu_feeder.Feeder_triple
train_feeder_args:                                            
  data_path: ./cobot_dataset_frame64/xsub/train_position.npy
  label_path: ./cobot_dataset_frame64/xsub/train_label.pkl
  shear_amplitude: 0.2
  temperal_padding_ratio: 6
  mmap: True
  aug_method: '1234'
  zero_z: True

# model
model: net.aimclr_v2_3views.AimCLR_v2_3views
model_args:
  base_encoder: net.st_gcn.Model
  pretrain: True
  feature_dim: 128
  queue_size: 8192
  momentum: 0.999
  Temperature: 0.2
  mlp: True
  in_channels: 3
  hidden_channels: 32
  hidden_dim: 256
  num_class: 19  
  dropout: 0.5
  graph_args:
    layout: 'cobot'
    strategy: 'distance'
  edge_importance_weighting: True

# optim
nesterov: False
weight_decay: 1e-4
base_lr: 0.05
optimizer: SGD
step: [120]

# training
device: [0]
batch_size: 32
test_batch_size: 32
num_epoch: 200
start_epoch: 0
stream: 'all'

# cross training
topk1: 4
topk2: 20
vote: 2
mine_epoch: 40
cross_epoch: 80

# log
save_interval: 10
eval_interval: -1
"""

with open('config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml', 'w') as f:
    f.write(pretext_config)
```

### Step 3: Process Data
```python
!python cleaned_to_ntu.py \
    --cleaned_root ./cleaned_actions \
    --out_root ./cobot_dataset_frame64 \
    --max_frame 64 \
    --resample uniform-sample \
    --train_ratio 0.8 \
    --seed 42
```

### Step 4: Start Training
```python
# Pretraining
!python main.py pretrain_aimclr_v2_3views \
    --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml

# Linear evaluation
!python main.py linear_evaluation \
    --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml
```

---

## 🔧 Troubleshooting

### Common Issues:

**1. CUDA Out of Memory (OOM)**
```python
# Reduce batch size in config files
batch_size: 16  # instead of 32
queue_size: 4096  # instead of 8192
```

**2. Import Errors**
- Make sure all Python files are uploaded
- Check file paths are correct
- Verify directory structure

**3. Dataset Issues**
- Ensure your dataset structure is correct
- Check action IDs are properly parsed
- Verify file permissions

**4. Training Stalls**
- Monitor GPU memory usage
- Reduce model complexity
- Check for infinite loops in code

### Performance Tips:

1. **Use Colab Pro** for longer sessions and better GPUs
2. **Monitor GPU memory** during training
3. **Save checkpoints** regularly
4. **Use smaller batch sizes** if memory is limited
5. **Enable GPU runtime** in Colab settings

---

## 📊 Monitoring Training

### Check Training Progress:
```python
# View pretraining logs
!tail -f work_dir/cobot_3views_2D_cleaned/pretext/log.txt

# View linear evaluation logs
!tail -f work_dir/cobot_3views_2D_cleaned/linear_eval/log.txt

# Check GPU usage
!nvidia-smi
```

### Download Results:
```python
# Create results archive
!zip -r aimclr_results.zip work_dir/ config/ cobot_dataset_frame64/

# Download
from google.colab import files
files.download('aimclr_results.zip')
```

---

## 🎯 Transfer Learning on Colab

To do transfer learning (pretrain on dataset A, fine-tune on dataset B):

1. **Upload source dataset** and run pretraining
2. **Upload target dataset** 
3. **Create fine-tuning config** with `unfreeze_backbone: True`
4. **Run fine-tuning** using the linear evaluation processor

Example fine-tuning config:
```yaml
weights: ./work_dir/source_dataset/pretext/epoch200_model.pt
unfreeze_backbone: True
base_lr: 0.01  # Lower LR for fine-tuning
```

---

## 📈 Expected Results

With proper setup, you should see:

- **Pretraining**: Loss decreasing over epochs
- **Linear Evaluation**: Top-1 accuracy improving
- **Final Results**: Best model saved as `best_model.pt`

Typical training time: 1-3 hours depending on dataset size.

---

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all files are uploaded correctly
3. Ensure GPU runtime is enabled
4. Check Colab's resource limits
5. Monitor GPU memory usage

**Happy Training! 🎉**
