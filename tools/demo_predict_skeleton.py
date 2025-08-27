#!/usr/bin/env python
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# Make sure we can import the NTU skeleton reader used by ntu_gendata
try:
    from utils.ntu_read_skeleton import read_xyz  # when running from repo root with script under tools/
except Exception:
    # Fallback: add tools/utils to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
    from ntu_read_skeleton import read_xyz

from net.aimclr_v2_3views import AimCLR_v2_3views


def center_crop_or_pad(data_ctvm: np.ndarray, t_length: int) -> np.ndarray:
    """Crop/pad along T to target length.
    data_ctvm: (C, T, V, M)
    """
    C, T, V, M = data_ctvm.shape
    if T == t_length:
        return data_ctvm
    if T > t_length:
        start = (T - t_length) // 2
        end = start + t_length
        return data_ctvm[:, start:end, :, :]
    out = np.zeros((C, t_length, V, M), dtype=data_ctvm.dtype)
    out[:, :T, :, :] = data_ctvm
    return out


def load_skeleton_as_ctvm(path: str, num_joint: int = 25, max_body: int = 2, t_length: int = 50) -> np.ndarray:
    ctvm = read_xyz(path, max_body=max_body, num_joint=num_joint)  # (C, T, V, M)
    ctvm = center_crop_or_pad(ctvm, t_length)
    return ctvm


def load_label_map(path: str):
    if not path:
        return None
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # expected format: index<space or tab>label
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                mapping[int(parts[0])] = parts[1]
    return mapping


def main():
    parser = argparse.ArgumentParser(description='Demo: predict action from a single NTU .skeleton file')
    parser.add_argument('--skeleton_file', type=str, required=True, help='Path to .skeleton file')
    parser.add_argument('--weights', type=str, required=True, help='Path to trained classifier weights (.pt) from linear eval')
    parser.add_argument('--layout', type=str, default='ntu-rgb+d', help="Graph layout: 'ntu-rgb+d' or 'cobot'")
    parser.add_argument('--num_class', type=int, default=60)
    parser.add_argument('--t_length', type=int, default=50, help='Temporal length to crop/pad to')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--class_map', type=str, default='', help='Optional label map txt: lines like "0 drink water"')
    args = parser.parse_args()

    assert os.path.isfile(args.skeleton_file), f'Missing skeleton file: {args.skeleton_file}'
    assert os.path.isfile(args.weights), f'Missing weights file: {args.weights}'

    # Set V and M based on layout
    if args.layout == 'cobot':
        num_joint, max_body = 48, 1
    else:
        num_joint, max_body = 25, 2

    # Load sample and prepare tensor
    ctvm = load_skeleton_as_ctvm(args.skeleton_file, num_joint=num_joint, max_body=max_body, t_length=args.t_length)
    # N,C,T,V,M
    nctvm = np.expand_dims(ctvm, 0).astype('float32')
    data = torch.from_numpy(nctvm)

    # Build model (classifier mode)
    model = AimCLR_v2_3views(
        base_encoder='net.st_gcn.Model',
        pretrain=False,
        in_channels=3,
        hidden_channels=16,
        hidden_dim=256,
        num_class=args.num_class,
        dropout=0.5,
        graph_args={'layout': args.layout, 'strategy': 'spatial'},
        edge_importance_weighting=True,
    )
    state = torch.load(args.weights, map_location='cpu')
    # Weight files usually contain 'model' or direct state_dict
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    model.load_state_dict(state, strict=False)

    device = torch.device(args.device)
    model.to(device).eval()
    data = data.to(device)

    with torch.no_grad():
        logits = model(None, data)  # stream='all' by default in non-pretrain
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()
        topk_idx = prob.argsort()[-args.topk:][::-1]

    label_map = load_label_map(args.class_map)
    print('Top-{} predictions:'.format(args.topk))
    for i in topk_idx:
        name = label_map.get(int(i), str(int(i))) if label_map else str(int(i))
        print(f'  {name}\t{prob[i]:.4f}')


if __name__ == '__main__':
    main()
