# Usage Examples for cleaned_to_ntu.py

## Cross-Subject Evaluation (Recommended for Action Recognition)

This is the proper way to evaluate action recognition models, as it tests generalization to unseen subjects.

```bash
python cleaned_to_ntu.py \
    --cleaned_root cleaned_microgaps_v2 \
    --out_root cobot_dataset_xsub \
    --split_strategy cross_subject \
    --max_frame 64 \
    --resample uniform-sample
```

This will:
- Split data by subject ID: odd subjects → training, even subjects → validation
- Create output in `cobot_dataset_xsub/xsub/` directory
- Ensure no subject overlap between train and validation sets
- Provide proper cross-subject evaluation protocol

## Random Split (For Comparison)

This is the original 80/20 random split for comparison purposes.

```bash
python cleaned_to_ntu.py \
    --cleaned_root cleaned_microgaps_v2 \
    --out_root cobot_dataset_random \
    --split_strategy random \
    --train_ratio 0.8 \
    --max_frame 64 \
    --resample uniform-sample
```

This will:
- Randomly split samples 80% train / 20% validation
- Create output in `cobot_dataset_random/random/` directory
- May have subject overlap between train and validation sets

## Key Differences

### Cross-Subject Split
- ✅ Tests generalization to unseen subjects
- ✅ More realistic evaluation scenario
- ✅ Follows standard action recognition protocols
- ✅ No subject overlap between train/val

### Random Split
- ❌ May have subject overlap between train/val
- ❌ Less realistic evaluation scenario
- ✅ Useful for debugging and development
- ✅ Balanced action distribution

## Output Structure

### Cross-Subject Evaluation
```
cobot_dataset_xsub/
└── xsub/
    ├── train_position.npy
    ├── train_label.pkl
    ├── val_position.npy
    └── val_label.pkl
```

### Random Split
```
cobot_dataset_random/
└── random/
    ├── train_position.npy
    ├── train_label.pkl
    ├── val_position.npy
    └── val_label.pkl
```

## Logging Output

The script provides detailed logging about:
- Number of samples found
- Subject distribution (for cross-subject split)
- Action distribution
- Split strategy used
- Output directory location

Example output for cross-subject split:
```
[INFO] Found 5000 samples across 20 actions
[INFO] Using cross-subject split: odd subjects -> train, even subjects -> val
[INFO] Train: 2500 samples, Val: 2500 samples
[INFO] Train subjects: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
[INFO] Val subjects: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
[INFO] Subject overlap: 0
```
