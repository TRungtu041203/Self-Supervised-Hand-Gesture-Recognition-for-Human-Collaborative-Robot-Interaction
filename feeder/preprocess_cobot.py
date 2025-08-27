import os
import sys
import pickle
import torch
import argparse
import numpy as np
from numpy.lib.format import open_memmap
from NTUDatasets import NTUMotionProcessor

max_body = 1  
num_joint = 48 
max_frame = 50  # Downsample to 50 frames
batch_size = 64

class COBOTMotionProcessor:
    """Custom processor for COBOT dataset with 48 joints"""
    
    def __init__(self, data_path, label_path, **kwargs):
        self.data_path = data_path
        self.label_path = label_path
        
        # Load data
        self.data = np.load(data_path, mmap_mode='r')
        with open(label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        
        self.N = len(self.label)
        self.t_length = max_frame
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, index):
        # Get data
        data_numpy = np.array(self.data[index])  # Shape: (3, T, 48, 1)
        label = self.label[index]
        
        # Handle variable length sequences
        T = data_numpy.shape[1]
        if T > self.t_length:
            # Center crop if too long
            start = (T - self.t_length) // 2
            end = start + self.t_length
            data_numpy = data_numpy[:, start:end, :, :]
        elif T < self.t_length:
            padded_data = np.zeros((3, self.t_length, 48, 1))
            padded_data[:, :T, :, :] = data_numpy
            data_numpy = padded_data
        
        motion_data = np.zeros_like(data_numpy)
        motion_data[:, :-1, :, :] = data_numpy[:, 1:, :, :] - data_numpy[:, :-1, :, :]
        
        return data_numpy, motion_data, label

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

def gendata(dataset_path, out_path, benchmark, part='eval'):
    """Generate position and motion data for COBOT dataset"""
    
    # Create dataset processor
    dataset = COBOTMotionProcessor(
        '{}/{}_data.npy'.format(os.path.join(dataset_path, benchmark), part),
        '{}/{}_label.pkl'.format(os.path.join(dataset_path, benchmark), part))

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
        drop_last=False)

    # Create output files
    f_position = open_memmap(
        '{}/{}_position.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(dataset.N, 3, max_frame, num_joint, max_body))

    f_motion = open_memmap(
        '{}/{}_motion.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(dataset.N, 3, max_frame, num_joint, max_body))

    f_label = open_memmap(
        '{}/{}_label.npy'.format(out_path, part),
        dtype='int64',
        mode='w+',
        shape=(dataset.N, 1))

    index = 0
    for i, (data, motion, label) in enumerate(data_loader):
        print_toolbar(i * 1.0 / len(data_loader),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(data_loader), benchmark, part))
        length = label.shape[0]
        if i * batch_size != index:
            print(i, index)
        f_position[index:(index+length), :, :, :, :] = data.numpy()
        f_motion[index:(index+length), :, :, :, :] = motion.numpy()
        f_label[index:(index+length), :] = label.numpy().reshape(-1, 1)
        index += length
    end_toolbar()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='COBOT Data Preprocessor.')
    parser.add_argument('--dataset_path', default='cobot_dataset')
    parser.add_argument('--out_folder', default='cobot_dataset_frame50')

    benchmark = ['xsub']  
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(arg.dataset_path, out_path, benchmark=b, part=p) 