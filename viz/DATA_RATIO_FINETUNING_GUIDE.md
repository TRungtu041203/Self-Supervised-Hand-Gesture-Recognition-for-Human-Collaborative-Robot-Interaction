# COBOT Data Ratio Finetuning Guide

## 🎯 **Overview**
This guide helps you finetune your COBOT model with different percentages of labeled training data to analyze the impact of training set size on performance.

## 📁 **Files Created**

### **1. New Feeder with Subset Support**
- **`feeder/ntu_feeder_subset.py`**: Enhanced feeder that supports training with data subsets
- **Features**:
  - Stratified sampling (maintains class distribution)
  - Reproducible results with random seed
  - Support for any data ratio (0.1 to 1.0)
  - Memory-efficient handling of large datasets

### **2. Configuration Files**
- **`config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_80percent.yaml`**: 80% data
- **`config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_60percent.yaml`**: 60% data  
- **`config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_40percent.yaml`**: 40% data

### **3. Execution Scripts**
- **`run_finetune_data_ratios.bat`**: Windows batch script to run all experiments
- **`run_finetune_data_ratios.sh`**: Linux shell script to run all experiments

### **4. Analysis Tools**
- **`analyze_data_ratio_results.py`**: Script to compare results across different data ratios

## 🚀 **How to Use**

### **Step 1: Run Single Experiment (80% data)**
```bash
# Windows
C:\Users\leeji\AppData\Local\Programs\Python\Python310\python.exe main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_80percent.yaml

# Linux
python main.py linear_evaluation --config config/cobot/finetune/finetune_aimclr_v2_3views_cobot_xsub_80percent.yaml
```

### **Step 2: Run All Experiments**
```bash
# Windows
run_finetune_data_ratios.bat

# Linux  
./run_finetune_data_ratios.sh
```

### **Step 3: Analyze Results**
```bash
python analyze_data_ratio_results.py
```

## 🔧 **Technical Details**

### **Stratified Sampling**
The subset feeder ensures balanced class representation:
```python
# Example output for 80% data:
📊 Applying data subset: 1200/1500 samples (80.0%)
  Class 0: 48/60 samples
  Class 1: 40/50 samples  
  Class 2: 32/40 samples
  ...
✅ Data subset applied: 1200 samples selected
```

### **Key Parameters**
```yaml
train_feeder_args:
  data_ratio: 0.8      # Use 80% of training data
  random_seed: 42      # For reproducible results
```

### **Work Directories**
Results are saved in separate directories:
- **100% data**: `work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune/`
- **80% data**: `work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune_80percent/`
- **60% data**: `work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune_60percent/`
- **40% data**: `work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune_40percent/`

## 📊 **Expected Results Analysis**

### **Performance vs Data Size**
Typical results might show:
```
Data Ratio    Best Acc    Final Acc    Status
100%          85.2%       84.8%        ✅ Complete
80%           83.1%       82.5%        ✅ Complete  
60%           79.8%       78.9%        ✅ Complete
40%           74.2%       73.1%        ✅ Complete
```

### **Key Insights**
- **Diminishing Returns**: Performance improvement slows with more data
- **Minimum Viable**: Find the smallest dataset that gives acceptable performance
- **Cost-Benefit**: Balance annotation cost vs performance gain
- **Robustness**: Test model stability with limited data

## 🎯 **Research Questions to Explore**

### **1. Data Efficiency**
- How much performance is lost with 80% vs 100% data?
- What's the minimum data needed for acceptable performance?

### **2. Learning Curves**
- Does the model overfit faster with less data?
- How does convergence speed change with data size?

### **3. Class-wise Performance**
- Which classes suffer most with reduced data?
- Are some actions more data-hungry than others?

### **4. Generalization**
- Does the model generalize better with more diverse (100%) vs focused (subset) data?

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"Feeder not found" error**:
   ```
   ModuleNotFoundError: No module named 'feeder.ntu_feeder_subset'
   ```
   **Solution**: Ensure `feeder/ntu_feeder_subset.py` exists and has correct imports

2. **"Pretrained model not found"**:
   ```
   FileNotFoundError: work_dir/cobot_3views_2D_xsub_medgap_aug1/pretext/epoch400_model.pt
   ```
   **Solution**: Run pretraining first or update the `weights` path in config files

3. **Memory issues with large datasets**:
   **Solution**: The subset feeder uses memory mapping (`mmap=True`) to handle large files efficiently

### **Validation**
Check that subset sampling works correctly:
```python
# The feeder should print:
📊 Applying data subset: X/Y samples (Z%)
  Class 0: a/b samples
  Class 1: c/d samples
  ...
✅ Data subset applied: X samples selected
```

## 📈 **Advanced Usage**

### **Custom Data Ratios**
Create your own config for any ratio:
```yaml
train_feeder_args:
  data_ratio: 0.3  # Use 30% of training data
  random_seed: 42
```

### **Different Random Seeds**
Test robustness with different data subsets:
```yaml
train_feeder_args:
  data_ratio: 0.8
  random_seed: 123  # Different subset selection
```

### **Class-specific Analysis**
Modify the feeder to focus on specific classes or exclude certain classes for targeted experiments.

## 🎉 **Expected Workflow**

1. **Run experiments**: Execute all data ratio experiments
2. **Compare results**: Use analysis script to generate comparison
3. **Visualize trends**: Review performance vs data size plots
4. **Make decisions**: Choose optimal data ratio for your use case
5. **Document findings**: Record insights for future reference

This setup provides a comprehensive framework for understanding how your COBOT model performs with varying amounts of training data! 🚀
