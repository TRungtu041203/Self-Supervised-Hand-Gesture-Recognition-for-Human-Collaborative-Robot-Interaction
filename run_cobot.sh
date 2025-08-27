#!/bin/bash

# COBOT Dataset Training Script for 3s-AimCLR++

echo "=== COBOT Dataset Integration for 3s-AimCLR++ ==="

# Step 1: Convert raw COBOT data
echo "Step 1: Converting raw COBOT data..."
python tools/cobot_gendata.py --data_path pose_clean --out_folder cobot_dataset --benchmark xsub

# Step 2: Preprocess data for 3-stream architecture
echo "Step 2: Preprocessing data for 3-stream architecture..."
python feeder/preprocess_cobot.py --dataset_path cobot_dataset --out_folder cobot_dataset_frame50

# Step 3: Pretrain 3s-AimCLR++ on COBOT
echo "Step 3: Pretraining 3s-AimCLR++ on COBOT dataset..."
python main.py pretrain_aimclr_v2_3views --config config/cobot/pretext/pretext_aimclr_v2_3views_cobot_xsub.yaml

# Step 4: Linear evaluation
echo "Step 4: Linear evaluation on COBOT dataset..."
python main.py linear_evaluation --config config/cobot/linear/linear_eval_aimclr_v2_3views_cobot_xsub.yaml

echo "=== COBOT Training Complete ===" 