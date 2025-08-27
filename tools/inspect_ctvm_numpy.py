#!/usr/bin/env python
import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Generic inspector for (N,C,T,V,M) numpy arrays.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to .npy file (e.g., train_position.npy)')
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    assert os.path.isfile(args.data_path), f'Missing file: {args.data_path}'
    data = np.load(args.data_path, mmap_mode='r')
    print('=== C-T-V-M numpy inspection ===')
    print(f'- path: {args.data_path}')
    print(f'- dtype: {data.dtype}')
    print(f'- shape: {data.shape}  # expected (N,C,T,V,M)')

    if data.ndim != 5:
        print('! Warning: array is not 5D (N,C,T,V,M)')
        return

    N, C, T, V, M = data.shape

    # basic stats over a subset
    ns = min(args.num_samples, N)
    mins = []
    maxs = []
    means = []
    stds = []
    for i in range(ns):
        x = np.array(data[i])
        mins.append(float(x.min()))
        maxs.append(float(x.max()))
        means.append(float(x.mean()))
        stds.append(float(x.std()))

    print(f'- subset size: {ns}/{N}')
    print(f'- coord_min (subset): {min(mins):.6f}')
    print(f'- coord_max (subset): {max(maxs):.6f}')
    print(f'- mean(mean) (subset): {np.mean(means):.6f}')
    print(f'- mean(std) (subset): {np.mean(stds):.6f}')


if __name__ == '__main__':
    main()
