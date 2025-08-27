#!/usr/bin/env python
import argparse
import os
import pickle
import random
import json
from typing import Tuple, List

import numpy as np


def load_labels(label_path: str) -> Tuple[List[str], np.ndarray]:
    with open(label_path, 'rb') as f:
        sample_name, label = pickle.load(f)
    labels = np.array(label, dtype=np.int64)
    return sample_name, labels


def effective_length(sample: np.ndarray) -> int:
    """Compute effective sequence length for one sample (C, T, V, M) by
    finding last frame with any non-zero value across persons and joints."""
    # sample shape: (C, T, V, M)
    # Reduce over C,V,M to get activity per frame
    frame_activity = np.abs(sample).sum(axis=(0, 2, 3))  # shape (T,)
    nonzero = np.where(frame_activity > 0)[0]
    return int(nonzero[-1] + 1) if nonzero.size > 0 else 0


def persons_present(sample: np.ndarray) -> int:
    """Count how many persons (M) are present (any non-zero)."""
    presence = (np.abs(sample).sum(axis=(0, 1, 2)) > 0)  # shape (M,)
    return int(presence.sum())


def sample_stats(data: np.memmap, indices: List[int]) -> dict:
    lengths = []
    persons = []
    mins = []
    maxs = []
    means = []
    stds = []
    for idx in indices:
        x = np.array(data[idx])  # (C,T,V,M)
        lengths.append(effective_length(x))
        persons.append(persons_present(x))
        mins.append(float(x.min()))
        maxs.append(float(x.max()))
        means.append(float(x.mean()))
        stds.append(float(x.std()))
    return {
        'effective_length_min': int(np.min(lengths)),
        'effective_length_max': int(np.max(lengths)),
        'effective_length_mean': float(np.mean(lengths)),
        'persons_present_min': int(np.min(persons)),
        'persons_present_max': int(np.max(persons)),
        'coord_min': float(np.min(mins)),
        'coord_max': float(np.max(maxs)),
        'coord_mean_mean': float(np.mean(means)),
        'coord_std_mean': float(np.mean(stds)),
    }


def main():
    parser = argparse.ArgumentParser(description='Inspect NTU gendata outputs (N,C,T,V,M) and labels.')
    parser.add_argument('--root', type=str, default='data_ntu/ntu60', help='Root folder where xsub/xview live')
    parser.add_argument('--benchmark', type=str, default='xsub', choices=['xsub', 'xview'])
    parser.add_argument('--part', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--num_samples', type=int, default=20, help='Number of random samples to summarize')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_json', type=str, default='', help='Optional path to save summary JSON')
    args = parser.parse_args()

    data_path = os.path.join(args.root, args.benchmark, f'{args.part}_data.npy')
    label_path = os.path.join(args.root, args.benchmark, f'{args.part}_label.pkl')

    assert os.path.isfile(data_path), f'Missing data file: {data_path}'
    assert os.path.isfile(label_path), f'Missing label file: {label_path}'

    data = np.load(data_path, mmap_mode='r')  # shape (N,C,T,V,M)
    N, C, T, V, M = data.shape
    sample_name, labels = load_labels(label_path)

    print('=== NTU gendata inspection ===')
    print(f'- data_path: {data_path}')
    print(f'- label_path: {label_path}')
    print(f'- shape: (N={N}, C={C}, T={T}, V={V}, M={M})')
    print(f'- labels: count={labels.size}, min={labels.min()}, max={labels.max()}')

    # label distribution (top 10 counts)
    uniq, cnt = np.unique(labels, return_counts=True)
    order = np.argsort(-cnt)
    print('- label distribution (top 10):')
    for i in order[:10]:
        print(f'  class {int(uniq[i])}: {int(cnt[i])}')

    # show some sample names/labels
    print('- examples:')
    for i in range(5):
        print(f'  [{i}] {sample_name[i]} -> label {int(labels[i])}')

    # random subset stats
    rng = random.Random(args.seed)
    indices = list(range(N))
    rng.shuffle(indices)
    sel = indices[: min(args.num_samples, N)]
    stats = sample_stats(data, sel)

    print('- sample stats (approx over subset):')
    for k, v in stats.items():
        print(f'  {k}: {v}')

    # effective length distribution approx
    lengths = []
    for idx in sel:
        lengths.append(effective_length(np.array(data[idx])))
    print(f'- effective length (subset) min/mean/max: {min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)} frames')

    out = {
        'shape': {'N': N, 'C': C, 'T': T, 'V': V, 'M': M},
        'labels': {
            'count': int(labels.size),
            'min': int(labels.min()),
            'max': int(labels.max()),
        },
        'subset_stats': stats,
        'examples': [{'name': sample_name[i], 'label': int(labels[i])} for i in range(min(5, N))],
    }

    if args.save_json:
        with open(args.save_json, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f'- saved summary to {args.save_json}')


if __name__ == '__main__':
    main()
