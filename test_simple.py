#!/usr/bin/env python3
"""
Simple test script for COBOT data loading
"""

import numpy as np

def test_basic_loading():
    """Test basic data loading"""
    try:
        # Test position data
        position = np.load('cobot_dataset_frame50/xsub/train_position.npy')
        print(f"Position data shape: {position.shape}")
        print(f"Position data range: [{position.min():.4f}, {position.max():.4f}]")
        
        # Test motion data
        motion = np.load('cobot_dataset_frame50/xsub/train_motion.npy')
        print(f"Motion data shape: {motion.shape}")
        print(f"Motion data range: [{motion.min():.4f}, {motion.max():.4f}]")
        
        # Test label data
        label = np.load('cobot_dataset_frame50/xsub/train_label.npy')
        print(f"Label data shape: {label.shape}")
        print(f"Labels: {np.unique(label)}")
        
        print("✅ Basic data loading successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == '__main__':
    test_basic_loading() 