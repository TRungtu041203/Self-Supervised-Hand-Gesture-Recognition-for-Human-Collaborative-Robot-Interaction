#!/bin/bash

echo "=== Running Finetune Evaluation for COBOT Dataset ==="
echo "This runs finetuning (unfreeze_backbone=True) on COBOT dataset"
echo ""

# Check if pretrained model exists
if [ ! -f "./work_dir/cobot_3views_2D_xsub_medgap/pretext/epoch400_model.pt" ]; then
    echo "Error: Pretrained model not found at ./work_dir/cobot_3views_2D_xsub_medgap/pretext/epoch400_model.pt"
    echo "Please run pretraining first or update the weights path in the config file."
    exit 1
fi

echo "Running finetune evaluation on COBOT xsub..."
python main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub.yaml

echo ""
echo "=== COBOT Finetune Evaluation Complete ==="
echo "Results are saved in: ./work_dir/cobot_3views_2D_xsub_medgap/finetune/"
