#!/usr/bin/env python3
"""
Google Colab Setup Script for AimCLR Training
=============================================

This script automates the setup and training of AimCLR on Google Colab.
Run this in a Colab notebook cell to set up everything automatically.

Usage in Colab:
1. Upload this script to Colab
2. Upload your preprocessed NTU-60 dataset (ntu60_frame50 folder as ZIP)
3. Upload your AimCLR code files
4. Run: !python colab_setup.py
5. Training will start automatically

Note: This script expects preprocessed NTU-60 data. Process your raw NTU-60 data locally first using ntu_gendata.py and preprocess_ntu.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            return True
        else:
            print("❌ No GPU detected! Please enable GPU runtime in Colab.")
            print("Runtime → Change runtime type → Hardware accelerator → GPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    packages = [
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "numpy scipy matplotlib tqdm pyyaml tensorboard",
        "opencv-python scikit-learn"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run(f"pip install {package}", shell=True, check=True)
    
    print("✅ Dependencies installed!")

def create_directory_structure():
    """Create the necessary directory structure"""
    print("📁 Creating directory structure...")
    
    directories = [
        "config/ntu60/pretext",
        "config/ntu60/linear", 
        "work_dir",
        "net",
        "processor",
        "feeder"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def create_config_files():
    """Create YAML configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Pretraining config
    pretext_config = """work_dir: ./work_dir/ntu60_cs/pretext

# feeder
train_feeder: feeder.ntu_feeder.Feeder_triple
train_feeder_args:                                            
  data_path: ./ntu60_frame50/xsub/train_position.npy
  label_path: ./ntu60_frame50/xsub/train_label.pkl
  shear_amplitude: 0.5
  temperal_padding_ratio: 6
  mmap: True
  aug_method: '12345'
  zero_z: False

# model
model: net.aimclr_v2_3views.AimCLR_v2_3views
model_args:
  base_encoder: net.st_gcn.Model
  pretrain: True
  feature_dim: 128
  queue_size: 32768
  momentum: 0.999
  Temperature: 0.07
  mlp: True
  in_channels: 3
  hidden_channels: 16 # 1/4 x channels of standard stgcn
  hidden_dim: 256
  num_class: 60
  dropout: 0.5
  graph_args:
    layout: 'ntu-rgb+d'
    strategy: 'spatial'
  edge_importance_weighting: True

# optim
nesterov: False
weight_decay: 1e-4
base_lr: 0.1
optimizer: SGD
step: [350]

# training
device: [0]
batch_size: 128
test_batch_size: 128
num_epoch: 400
start_epoch: 0
stream: 'all'

# cross training
topk1: 4
topk2: 20
vote: 2
mine_epoch: 150
cross_epoch: 300

# log
save_interval: 5
eval_interval: -1
"""
    
    # Linear evaluation config
    linear_config = """work_dir: ./work_dir/ntu60_cs/linear_eval

# initialize from your pretraining checkpoint
weights: ./work_dir/ntu60_cs/pretext/epoch400_model.pt
ignore_weights: [encoder_k, encoder_k_motion, encoder_k_bone, queue, queue_motion, queue_bone]

# feeder
train_feeder: feeder.ntu_feeder.Feeder_single
train_feeder_args:
  data_path: ./ntu60_frame50/xsub/train_position.npy
  label_path: ./ntu60_frame50/xsub/train_label.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True
  zero_z: False
test_feeder: feeder.ntu_feeder.Feeder_single
test_feeder_args:
  data_path: ./ntu60_frame50/xsub/val_position.npy
  label_path: ./ntu60_frame50/xsub/val_label.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True
  zero_z: False

# model
model: net.aimclr_v2_3views.AimCLR_v2_3views
model_args:
  base_encoder: net.st_gcn.Model
  pretrain: False
  in_channels: 3
  hidden_channels: 16
  hidden_dim: 256
  num_class: 60
  dropout: 0.5
  graph_args:
    layout: 'ntu-rgb+d'
    strategy: 'spatial'
  edge_importance_weighting: True

# optim
nesterov: False
weight_decay: 0.0
base_lr: 3.0
optimizer: SGD
step: [80]

# training
device: [0]
batch_size: 128
test_batch_size: 128
num_epoch: 100
stream: 'joint'

# log
save_interval: -1
eval_interval: 5
"""
    
    # Write config files
    with open('config/ntu60/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml', 'w') as f:
        f.write(pretext_config)
    
    with open('config/ntu60/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml', 'w') as f:
        f.write(linear_config)
    
    print("✅ Configuration files created!")

def check_uploaded_files():
    """Check if required files are uploaded"""
    print("📋 Checking uploaded files...")
    
    required_files = [
        'main.py',
        'net/aimclr_v2_3views.py',
        'processor/linear_evaluation.py',
        'feeder/ntu_feeder.py',
        'feeder/tools.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📤 Please upload the missing files using the file upload widget.")
        return False
    else:
        print("✅ All required files found!")
        return True

def check_preprocessed_data():
    """Check if preprocessed dataset is uploaded"""
    print("📊 Checking preprocessed dataset...")
    
    # Check for the processed data files
    required_files = [
        'ntu60_frame50/xsub/train_position.npy',
        'ntu60_frame50/xsub/val_position.npy',
        'ntu60_frame50/xsub/train_label.pkl',
        'ntu60_frame50/xsub/val_label.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing preprocessed data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n📤 Please upload your preprocessed NTU-60 dataset (ntu60_frame50 folder as ZIP).")
        print("Expected structure:")
        print("ntu60_frame50/")
        print("└── xsub/")
        print("    ├── train_position.npy")
        print("    ├── val_position.npy")
        print("    ├── train_label.pkl")
        print("    └── val_label.pkl")
        return False
    else:
        print("✅ Preprocessed NTU-60 dataset found!")
        
        # Show data statistics
        try:
            import numpy as np
            train_data = np.load('ntu60_frame50/xsub/train_position.npy')
            val_data = np.load('ntu60_frame50/xsub/val_position.npy')
            print(f"📈 Dataset statistics:")
            print(f"   - Train samples: {train_data.shape[0]}")
            print(f"   - Val samples: {val_data.shape[0]}")
            print(f"   - Data shape: {train_data.shape[1:]}")
            print(f"   - Memory usage: {train_data.nbytes / 1024**2:.1f} MB")
        except Exception as e:
            print(f"⚠️ Could not load data statistics: {e}")
        
        return True

def start_training():
    """Start the training process"""
    print("🚀 Starting training...")
    print("This will take 1-3 hours depending on your dataset size.")
    
    try:
        # Start pretraining
        print("\n📚 Phase 1: Pretraining...")
        subprocess.run([
            "python", "main.py", "pretrain_aimclr_v2_3views",
            "--config", "config/ntu60/pretext/pretext_aimclr_v2_3views_ntu60_xsub.yaml"
        ], check=True)
        
        print("✅ Pretraining completed!")
        
        # Start linear evaluation
        print("\n🔍 Phase 2: Linear Evaluation...")
        subprocess.run([
            "python", "main.py", "linear_evaluation",
            "--config", "config/ntu60/linear/linear_eval_aimclr_v2_3views_ntu60_xsub.yaml"
        ], check=True)
        
        print("✅ Linear evaluation completed!")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during training: {e}")
        return False

def show_results():
    """Show training results"""
    print("📊 Training Results:")
    
    # Show pretraining log
    if Path("work_dir/ntu60_cs/pretext/log.txt").exists():
        print("\n=== Pretraining Log (last 10 lines) ===")
        subprocess.run("tail -10 work_dir/ntu60_cs/pretext/log.txt", shell=True)
    
    # Show linear evaluation log
    if Path("work_dir/ntu60_cs/linear_eval/log.txt").exists():
        print("\n=== Linear Evaluation Log (last 10 lines) ===")
        subprocess.run("tail -10 work_dir/ntu60_cs/linear_eval/log.txt", shell=True)
    
    # Show model files
    print("\n=== Model Files ===")
    subprocess.run("find work_dir/ -name '*.pt' -type f", shell=True)

def create_download_script():
    """Create a script to download results"""
    download_script = """#!/bin/bash
# Download results script
echo "📥 Creating results archive..."
zip -r aimclr_results.zip work_dir/ config/ ntu60_frame50/
echo "✅ Results archived as aimclr_results.zip"
echo "📤 Use the download button in Colab to get your results!"
"""
    
    with open('download_results.sh', 'w') as f:
        f.write(download_script)
    
    subprocess.run("chmod +x download_results.sh", shell=True)
    print("✅ Download script created! Run: !bash download_results.sh")

def main():
    """Main setup function"""
    print("🎯 AimCLR Colab Setup")
    print("=" * 50)
    
    # Step 1: Check GPU
    if not check_gpu():
        return
    
    # Step 2: Install dependencies
    install_dependencies()
    
    # Step 3: Create directory structure
    create_directory_structure()
    
    # Step 4: Create config files
    create_config_files()
    
    # Step 5: Check uploaded files
    if not check_uploaded_files():
        print("\n📋 Please upload the missing files and run this script again.")
        return
    
    # Step 6: Check preprocessed dataset
    if not check_preprocessed_data():
        print("\n📋 Please upload your preprocessed dataset and run this script again.")
        return
    
    # Step 7: Start training
    if not start_training():
        print("\n❌ Training failed. Check the error messages above.")
        return
    
    # Step 8: Show results
    show_results()
    
    # Step 9: Create download script
    create_download_script()
    
    print("\n🎉 Setup and training completed successfully!")
    print("📥 Run '!bash download_results.sh' to prepare your results for download.")

if __name__ == "__main__":
    main()
