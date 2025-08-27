#!/usr/bin/env python3
"""
Convert label files from .npy to .pkl format for COBOT dataset
"""

import numpy as np
import pickle
import os

def convert_labels():
    """Convert label files from .npy to .pkl format"""
    
    base_path = 'cobot_dataset_frame50/xsub'
    
    # Convert train labels
    train_labels = np.load(f'{base_path}/train_label.npy')
    train_sample_names = [f'train_sample_{i}' for i in range(len(train_labels))]
    
    with open(f'{base_path}/train_label.pkl', 'wb') as f:
        pickle.dump((train_sample_names, train_labels.flatten().tolist()), f)
    
    print(f"Converted train labels: {len(train_labels)} samples")
    
    # Convert val labels
    val_labels = np.load(f'{base_path}/val_label.npy')
    val_sample_names = [f'val_sample_{i}' for i in range(len(val_labels))]
    
    with open(f'{base_path}/val_label.pkl', 'wb') as f:
        pickle.dump((val_sample_names, val_labels.flatten().tolist()), f)
    
    print(f"Converted val labels: {len(val_labels)} samples")
    
    print("Label conversion completed!")

if __name__ == '__main__':
    convert_labels() 