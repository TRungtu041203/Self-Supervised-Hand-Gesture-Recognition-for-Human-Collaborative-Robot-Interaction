#!/usr/bin/env python3
"""
Debug script to check COBOT data for issues
"""

import numpy as np
import torch

def check_data():
    """Check COBOT data for potential issues"""
    
    print("=== COBOT Data Debug ===")
    
    # Load data
    position_data = np.load('cobot_dataset_frame50/xsub/train_position.npy')
    motion_data = np.load('cobot_dataset_frame50/xsub/train_motion.npy')
    label_data = np.load('cobot_dataset_frame50/xsub/train_label.npy')
    
    print(f"Position data shape: {position_data.shape}")
    print(f"Motion data shape: {motion_data.shape}")
    print(f"Label data shape: {label_data.shape}")
    
    # Check for NaN or Inf values
    print(f"\nPosition data - NaN: {np.isnan(position_data).sum()}, Inf: {np.isinf(position_data).sum()}")
    print(f"Motion data - NaN: {np.isnan(motion_data).sum()}, Inf: {np.isinf(motion_data).sum()}")
    print(f"Label data - NaN: {np.isnan(label_data).sum()}, Inf: {np.isinf(label_data).sum()}")
    
    # Check value ranges
    print(f"\nPosition data range: [{position_data.min():.6f}, {position_data.max():.6f}]")
    print(f"Motion data range: [{motion_data.min():.6f}, {motion_data.max():.6f}]")
    print(f"Label data range: [{label_data.min()}, {label_data.max()}]")
    
    # Check for zero variance
    position_std = position_data.std(axis=(1, 2, 3, 4))
    motion_std = motion_data.std(axis=(1, 2, 3, 4))
    
    print(f"\nPosition std - min: {position_std.min():.6f}, max: {position_std.max():.6f}")
    print(f"Motion std - min: {motion_std.min():.6f}, max: {motion_std.max():.6f}")
    
    # Check for samples with zero variance
    zero_var_position = (position_std == 0).sum()
    zero_var_motion = (motion_std == 0).sum()
    
    print(f"\nSamples with zero variance - Position: {zero_var_position}, Motion: {zero_var_motion}")
    
    # Test data loading with torch
    print("\n=== Testing PyTorch Data Loading ===")
    
    try:
        # Convert to torch tensors
        position_tensor = torch.from_numpy(position_data).float()
        motion_tensor = torch.from_numpy(motion_data).float()
        label_tensor = torch.from_numpy(label_data).long()
        
        print(f"Position tensor shape: {position_tensor.shape}")
        print(f"Motion tensor shape: {motion_tensor.shape}")
        print(f"Label tensor shape: {label_tensor.shape}")
        
        # Check for NaN in tensors
        print(f"Position tensor - NaN: {torch.isnan(position_tensor).sum().item()}")
        print(f"Motion tensor - NaN: {torch.isnan(motion_tensor).sum().item()}")
        
        # Test normalization
        position_mean = position_tensor.mean()
        position_std = position_tensor.std()
        
        print(f"Position tensor - Mean: {position_mean.item():.6f}, Std: {position_std.item():.6f}")
        
        # Test a simple forward pass
        print("\n=== Testing Simple Forward Pass ===")
        
        # Take a small batch
        batch_size = 4
        batch_position = position_tensor[:batch_size]
        batch_motion = motion_tensor[:batch_size]
        batch_label = label_tensor[:batch_size]
        
        print(f"Batch position shape: {batch_position.shape}")
        print(f"Batch motion shape: {batch_motion.shape}")
        print(f"Batch label shape: {batch_label.shape}")
        
        # Test bone computation
        bone_connections = [
            (43, 44), (44, 45), (45, 46), (46, 47), (47, 48),
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
            (17, 18), (18, 19), (19, 20), (20, 21),
            (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31),
            (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
            (38, 39), (39, 40), (40, 41), (41, 42),
            (21, 43), (42, 48),
        ]
        
        bone_data = torch.zeros_like(batch_position)
        for v1, v2 in bone_connections:
            v1_idx = v1 - 1
            v2_idx = v2 - 1
            bone_data[:, :, :, v1_idx, :] = batch_position[:, :, :, v1_idx, :] - batch_position[:, :, :, v2_idx, :]
        
        print(f"Bone data shape: {bone_data.shape}")
        print(f"Bone data - NaN: {torch.isnan(bone_data).sum().item()}")
        print(f"Bone data range: [{bone_data.min().item():.6f}, {bone_data.max().item():.6f}]")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    check_data() 