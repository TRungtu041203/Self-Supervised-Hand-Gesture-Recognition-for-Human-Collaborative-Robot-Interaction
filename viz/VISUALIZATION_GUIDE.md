# COBOT Dataset Visualization and Analysis Guide

This guide explains how to use the t-SNE visualization and NMI calculation tools for your COBOT dataset.

## Overview

The visualization tools provide:
- **t-SNE Visualization**: 2D visualization of learned features
- **NMI Calculation**: Normalized Mutual Information score for clustering quality
- **Feature Analysis**: Comprehensive analysis of learned representations
- **Class Separation Analysis**: Analysis of how well classes are separated

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements_visualization.txt
```

### 2. Required Files

Make sure you have:
- **Trained Model**: `./work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/epoch250_model.pt`
- **Data**: `./cobot_med_frame64/xsub/val_position.npy`
- **Labels**: `./cobot_med_frame64/xsub/val_label.pkl`

## Quick Start

### Method 1: Simple Script (Recommended)

```bash
python run_visualization_analysis.py
```

This will:
1. Check if all required files exist
2. Load your trained model
3. Extract features from 1000 samples
4. Generate t-SNE visualization
5. Calculate NMI score
6. Create comprehensive analysis plots
7. Save results to `./visualization_results/`

### Method 2: Command Line Tool

```bash
python tools/visualization_analysis.py \
    --model_path ./work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/epoch250_model.pt \
    --data_path ./cobot_med_frame64/xsub/val_position.npy \
    --label_path ./cobot_med_frame64/xsub/val_label.pkl \
    --num_classes 19 \
    --output_dir ./visualization_results \
    --max_samples 1000
```

### Method 3: Interactive Python

```python
from tools.visualization_analysis import COBOTVisualizationAnalyzer

# Create analyzer
analyzer = COBOTVisualizationAnalyzer(
    model_path='./work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/epoch250_model.pt',
    data_path='./cobot_med_frame64/xsub/val_position.npy',
    label_path='./cobot_med_frame64/xsub/val_label.pkl',
    num_classes=19
)

# Load model and data
analyzer.load_model_and_data()

# Extract features
features, labels = analyzer.extract_features(max_samples=1000)

# Compute t-SNE
tsne_embedding = analyzer.compute_tsne()

# Compute NMI
nmi_score, cluster_labels = analyzer.compute_nmi()

# Generate plots
analyzer.plot_tsne()
analyzer.plot_feature_statistics()
analyzer.plot_class_separation_analysis()

print(f"NMI Score: {nmi_score:.4f}")
```

## Output Files

The analysis generates several files in the output directory:

### 1. Visualizations
- **`tsne_visualization.png`**: t-SNE plot showing feature clusters
- **`feature_statistics.png`**: Feature distribution and statistics
- **`class_separation_analysis.png`**: Class separation analysis

### 2. Data Files
- **`analysis_results.pkl`**: Numerical results (NMI score, statistics)
- **`analysis_report.txt`**: Text report with detailed statistics

## Understanding the Results

### t-SNE Visualization

The t-SNE plot shows:
- **Each point**: Represents one action sample
- **Color**: Indicates the action class
- **Clustering**: Well-separated clusters indicate good feature learning
- **Overlap**: Overlapping clusters suggest similar actions or poor separation

### NMI Score

**Normalized Mutual Information (NMI)** measures clustering quality:
- **Range**: 0.0 to 1.0
- **0.0**: Random clustering (worst)
- **1.0**: Perfect clustering (best)
- **Good scores**: > 0.5
- **Excellent scores**: > 0.7

### Feature Statistics

The feature analysis shows:
- **Feature Distribution**: How feature values are distributed
- **Feature Variance**: Which features are most important
- **Class Distribution**: Balance of samples across classes
- **Feature Correlations**: Relationships between features

### Class Separation Analysis

This analysis provides:
- **Inter-class Distances**: How far apart different classes are
- **Intra-class Variance**: How compact each class is
- **Silhouette Score**: Overall clustering quality
- **Feature Importance**: Which features contribute most to separation

## Interpreting Results

### Good Results
- **High NMI score** (> 0.6)
- **Well-separated clusters** in t-SNE
- **Low intra-class variance**
- **High inter-class distances**

### Poor Results
- **Low NMI score** (< 0.3)
- **Overlapping clusters** in t-SNE
- **High intra-class variance**
- **Low inter-class distances**

## Troubleshooting

### Common Issues

1. **File Not Found Errors**
   ```
   ❌ Missing: ./work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/epoch250_model.pt
   ```
   **Solution**: Update paths in `run_visualization_analysis.py` or ensure files exist

2. **Memory Issues**
   ```
   CUDA out of memory
   ```
   **Solution**: Reduce `max_samples` in the configuration

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution**: Install dependencies: `pip install -r requirements_visualization.txt`

4. **Model Loading Errors**
   ```
   RuntimeError: Error(s) in loading state_dict
   ```
   **Solution**: Ensure model architecture matches training configuration

### Performance Tips

1. **Reduce Sample Count**: Use `max_samples=500` for faster processing
2. **Use CPU**: Set `device='cpu'` if GPU memory is limited
3. **Batch Processing**: Increase `batch_size` for faster feature extraction

## Customization

### Modify Action Names

Update the action names in `tools/visualization_analysis.py`:

```python
self.action_names = [
    'Your_Action_1', 'Your_Action_2', 'Your_Action_3',
    # ... add all your action names
]
```

### Change Visualization Style

Modify plot parameters:

```python
# Change t-SNE parameters
tsne_embedding = analyzer.compute_tsne(
    perplexity=50,  # Default: 30
    n_iter=2000,    # Default: 1000
    random_state=123
)

# Change plot style
analyzer.plot_tsne(figsize=(15, 12))
```

### Analyze Different Models

Compare different models:

```python
# Linear evaluation model
analyzer_linear = COBOTVisualizationAnalyzer(
    model_path='./work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/epoch250_model.pt',
    # ...
)

# Finetune model
analyzer_finetune = COBOTVisualizationAnalyzer(
    model_path='./work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune/epoch100_model.pt',
    # ...
)

# Compare NMI scores
nmi_linear = analyzer_linear.compute_nmi()[0]
nmi_finetune = analyzer_finetune.compute_nmi()[0]
print(f"Linear NMI: {nmi_linear:.4f}")
print(f"Finetune NMI: {nmi_finetune:.4f}")
```

## Advanced Usage

### Multi-Stream Analysis

Compare different streams:

```python
# Joint stream only
features_joint = analyzer.model(None, data, stream='joint')

# Motion stream only  
features_motion = analyzer.model(None, data, stream='motion')

# Bone stream only
features_bone = analyzer.model(None, data, stream='bone')

# Three-stream fusion
features_all = analyzer.model(None, data, stream='all')
```

### Temporal Analysis

Analyze features over time:

```python
# Extract features from different time steps
features_temporal = []
for t in range(data.shape[2]):  # Time dimension
    data_t = data[:, :, t:t+1, :, :]  # Single time step
    features_t = analyzer.model(None, data_t, stream='all')
    features_temporal.append(features_t)
```

## Expected Results

For a well-trained COBOT model, you should expect:

- **NMI Score**: 0.6 - 0.8
- **t-SNE**: Clear separation between action classes
- **Feature Variance**: Good distribution of important features
- **Class Separation**: Low intra-class variance, high inter-class distances

## Notes

- The analysis uses **three-stream fusion** features by default
- **1000 samples** are used by default for faster processing
- Results are saved in **high-resolution PNG format**
- **NMI calculation** uses K-means clustering with the same number of clusters as classes
- The tool automatically handles **GPU/CPU** selection based on availability
