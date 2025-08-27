#!/usr/bin/env python3
"""
COBOT Dataset Visualization and Analysis Script

This script runs t-SNE visualization and NMI calculation for your COBOT dataset.
It analyzes the learned features from your trained model.

Usage:
    python run_visualization_analysis.py
"""

import os
import numpy as np
from tools.visualization_analysis import COBOTVisualizationAnalyzer

def main():
    print("=== COBOT Dataset Visualization and Analysis ===")
    print("This will generate t-SNE plots (with NMI) and a feature correlation matrix")
    print()

    # Configuration - Update these paths based on your setup
    config = {
        'model_path': './work_dir/cobot_3views_2D_xsub_medgap_aug1/pretext/epoch400_model.pt',
        'data_path': './cobot_med_frame64/xsub/val_position.npy',
        'label_path': './cobot_med_frame64/xsub/val_label.pkl',
        'num_classes': 19,
        'output_dir': './visualization_results_pre',
        'max_samples': 1262  # Limit for faster processing
    }

    # Check if files exist
    print("Checking required files...")
    missing_files = []
    for key, path in config.items():
        if key in ['model_path', 'data_path', 'label_path']:
            if not os.path.exists(path):
                missing_files.append(path)
                print(f"❌ Missing: {path}")
            else:
                print(f"✅ Found: {path}")

    if missing_files:
        print("\n❌ Some required files are missing!")
        print("Please update the paths in this script or ensure the files exist.")
        print("\nExpected file locations:")
        print("- Model: ./work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune/best_model.pt")
        print("- Data:  ./cobot_med_frame64/xsub/val_position.npy")
        print("- Labels: ./cobot_med_frame64/xsub/val_label.pkl")
        return

    print("\n✅ All files found! Starting analysis...\n")

    # Create analyzer
    analyzer = COBOTVisualizationAnalyzer(
        model_path=config['model_path'],
        data_path=config['data_path'],
        label_path=config['label_path'],
        num_classes=config['num_classes']
    )

    # Load model and data
    analyzer.load_model_and_data()

    # Extract features (limited for faster processing)
    print(f"\nExtracting features from {config['max_samples']} samples...")
    features, labels = analyzer.extract_features(max_samples=config['max_samples'])

    # Compute t-SNE
    print("\nComputing t-SNE embedding...")
    tsne_embedding = analyzer.compute_tsne()

    # Compute NMI
    print("\nComputing NMI score...")
    nmi_score, cluster_labels = analyzer.compute_nmi()

    # Generate plots
    print("\nGenerating visualizations...")
    os.makedirs(config['output_dir'], exist_ok=True)

    # t-SNE plot (with NMI in the top-right corner)
    analyzer.plot_tsne(
        save_path=os.path.join(config['output_dir'], 'tsne_visualization.png'),
        nmi_score=nmi_score
    )

    # Feature correlation matrix (first 100 features by default)
    analyzer.plot_feature_correlation(
        save_path=os.path.join(config['output_dir'], 'feature_correlation.png'),
        sample_size=100
    )

    # Print results
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    print(f"NMI Score: {nmi_score:.4f}")
    print(f"Number of samples analyzed: {len(labels)}")
    print(f"Number of classes: {len(set(labels))}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Results saved to: {config['output_dir']}")

    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique_labels, counts):
        action_name = analyzer.action_names[label] if label < len(analyzer.action_names) else f"Action {label}"
        print(f"  {action_name}: {count} samples")

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"📊 NMI Score: {nmi_score:.4f}")
    print(f"📁 Results saved to: {config['output_dir']}")
    print("📈 Generated plots:")
    print("   - t-SNE visualization: tsne_visualization.png")
    print("   - Feature correlation: feature_correlation.png")

if __name__ == '__main__':
    main()
