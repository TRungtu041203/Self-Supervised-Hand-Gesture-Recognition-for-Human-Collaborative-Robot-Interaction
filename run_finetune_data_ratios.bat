@echo off
echo ========================================
echo COBOT Finetune with Different Data Ratios
echo ========================================

set PYTHON_CMD=C:\Users\leeji\AppData\Local\Programs\Python\Python310\python.exe

echo.
echo Checking for pretrained model...
if not exist "work_dir\cobot_3views_2D_xsub_medgap_aug1\pretext\epoch400_model.pt" (
    echo ❌ Error: Pretrained model not found!
    echo Please ensure you have run pretraining first and the model exists at:
    echo work_dir\cobot_3views_2D_xsub_medgap_aug1\pretext\epoch400_model.pt
    pause
    exit /b 1
)

echo ✅ Pretrained model found!
echo.

echo ========================================
echo 1. Finetuning with 100%% of labeled data
echo ========================================
echo Starting: %date% %time%
%PYTHON_CMD% main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub.yaml
if %errorlevel% neq 0 (
    echo ❌ Error in 100%% data finetuning
    pause
    exit /b 1
)
echo ✅ 100%% data finetuning completed!
echo.

echo ========================================
echo 2. Finetuning with 80%% of labeled data
echo ========================================
echo Starting: %date% %time%
%PYTHON_CMD% main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_80percent.yaml
if %errorlevel% neq 0 (
    echo ❌ Error in 80%% data finetuning
    pause
    exit /b 1
)
echo ✅ 80%% data finetuning completed!
echo.

echo ========================================
echo 3. Finetuning with 60%% of labeled data
echo ========================================
echo Starting: %date% %time%
%PYTHON_CMD% main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_60percent.yaml
if %errorlevel% neq 0 (
    echo ❌ Error in 60%% data finetuning
    pause
    exit /b 1
)
echo ✅ 60%% data finetuning completed!
echo.

echo ========================================
echo 4. Finetuning with 40%% of labeled data
echo ========================================
echo Starting: %date% %time%
%PYTHON_CMD% main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_40percent.yaml
if %errorlevel% neq 0 (
    echo ❌ Error in 40%% data finetuning
    pause
    exit /b 1
)
echo ✅ 40%% data finetuning completed!
echo.

echo ========================================
echo 🎉 All finetuning experiments completed!
echo ========================================
echo.
echo Results saved in:
echo - work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune\ (100%% data)
echo - work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune_80percent\
echo - work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune_60percent\
echo - work_dir\cobot_3views_2D_xsub_medgap_aug1\finetune_40percent\
echo.
echo Completed: %date% %time%
pause
