import os
import sys
import pickle
import argparse
import numpy as np
from numpy.lib.format import open_memmap
import re

# COBOT dataset parameters
max_body = 1  # Single person
num_joint = 48  # COBOT has 48 joints
max_frame = 300  # Maximum frames to process
toolbar_width = 30

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")

def parse_filename(filename):
    """Parse COBOT filename to extract subject, person, and action info"""
    # Format: s{subject_id}_{person_name}_{action_id}.npy
    match = re.match(r's(\d+)_(.+)_(\d+)\.npy', filename)
    if match:
        subject_id = int(match.group(1))
        person_name = match.group(2)
        action_id = int(match.group(3))
        return subject_id, person_name, action_id
    return None, None, None

def load_cobot_data(file_path):
    """Load COBOT data and convert to standard format"""
    # Load data: (T, V, C) -> (C, T, V, M)
    data = np.load(file_path)
    T, V, C = data.shape
    
    # Convert to (C, T, V, M) format
    data_converted = np.zeros((C, T, V, max_body))
    data_converted[:, :, :, 0] = data.transpose(2, 0, 1)
    
    return data_converted

def gendata(data_path, out_path, benchmark='xsub', part='train'):
    """Generate COBOT dataset in 3s-AimCLR++ format"""
    
    # Get all .npy files
    files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    files.sort()
    
    sample_name = []
    sample_label = []
    
    # Define training subjects (you can modify this based on your needs)
    if benchmark == 'xsub':
        # Cross-subject: use first 70% of subjects for training
        all_subjects = list(set([parse_filename(f)[0] for f in files if parse_filename(f)[0] is not None]))
        all_subjects.sort()
        split_idx = int(len(all_subjects) * 0.7)
        training_subjects = all_subjects[:split_idx]
    elif benchmark == 'xview':
        # Cross-view: use first 2 actions for training (assuming 3 actions per person)
        training_actions = [1, 2]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    
    print(f"Processing {len(files)} files...")
    print(f"Benchmark: {benchmark}")
    
    for filename in files:
        subject_id, person_name, action_id = parse_filename(filename)
        
        if subject_id is None:
            print(f"Warning: Could not parse filename {filename}")
            continue
            
        # Determine if this sample belongs to train/val split
        if benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xview':
            istraining = (action_id in training_actions)
        
        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not istraining
        else:
            raise ValueError(f"Unknown part: {part}")
        
        if issample:
            sample_name.append(filename)
            # Use action_id as label (assuming actions are 1-indexed)
            sample_label.append(action_id - 1)  # Convert to 0-indexed
    
    print(f"Found {len(sample_name)} samples for {part} set")
    
    # Save labels
    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    
    # Create data array
    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 3, max_frame, num_joint, max_body))
    
    # Process each file
    for i, filename in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        
        file_path = os.path.join(data_path, filename)
        data = load_cobot_data(file_path)
        
        # Handle variable length sequences
        T = data.shape[1]
        if T > max_frame:
            # Center crop if too long
            start = (T - max_frame) // 2
            end = start + max_frame
            data = data[:, start:end, :, :]
        elif T < max_frame:
            # Pad with zeros if too short
            padded_data = np.zeros((3, max_frame, num_joint, max_body))
            padded_data[:, :T, :, :] = data
            data = padded_data
        
        fp[i, :, :, :, :] = data
    
    end_toolbar()
    print(f"Saved {len(sample_name)} samples to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COBOT Data Converter.')
    parser.add_argument('--data_path', default='pose_clean')
    parser.add_argument('--out_folder', default='cobot_dataset')
    parser.add_argument('--benchmark', default='xsub', choices=['xsub', 'xview'])
    
    arg = parser.parse_args()
    
    # Create output directory
    if not os.path.exists(arg.out_folder):
        os.makedirs(arg.out_folder)
    
    # Generate data for both train and val splits
    for part in ['train', 'val']:
        out_path = os.path.join(arg.out_folder, arg.benchmark)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        
        gendata(
            arg.data_path,
            out_path,
            benchmark=arg.benchmark,
            part=part) 