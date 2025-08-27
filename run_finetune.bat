@echo off
echo === Running Finetune Evaluation for AimCLR-v2 ===
echo This runs finetuning (unfreeze_backbone=True) on NTU60 dataset
echo.

REM Check if pretrained model exists
if not exist ".\work_dir\ntu60_cs\pretext\epoch15_model.pt" (
    echo Error: Pretrained model not found at .\work_dir\ntu60_cs\pretext\epoch15_model.pt
    echo Please run pretraining first or update the weights path in the config file.
    pause
    exit /b 1
)

if not exist ".\work_dir\ntu60_cv\pretext\epoch15_model.pt" (
    echo Error: Pretrained model not found at .\work_dir\ntu60_cv\pretext\epoch15_model.pt
    echo Please run pretraining first or update the weights path in the config file.
    pause
    exit /b 1
)

echo Running finetune evaluation on NTU60 xsub...
python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xsub.yaml

echo.
echo Running finetune evaluation on NTU60 xview...
python main.py linear_evaluation --config config/three-stream/finetune/finetune_aimclr_v2_3views_ntu60_xview.yaml

echo.
echo === Finetune Evaluation Complete ===
echo Results are saved in:
echo - NTU60 xsub: .\work_dir\ntu60_cs\finetune\
echo - NTU60 xview: .\work_dir\ntu60_cv\finetune\
pause
