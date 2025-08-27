#!/usr/bin/env python3
"""
Enhanced COBOT Dataset Visualization and Analysis Script

This script runs enhanced t-SNE visualization and NMI calculation for your COBOT dataset
with better clustering, outlier removal, and class balancing.

Usage:
    python run_enhanced_visualization.py [--model_type MODEL_TYPE]
"""

import os
import sys
import argparse
from tools.visualization_analysis_enhanced import EnhancedCOBOTVisualizationAnalyzer

def check_model_files():
    """Check which model files are available"""
    model_files = {
        'pretrain': './work_dir/cobot_3views_2D_xsub_medgap_aug1/pretext/epoch400_model.pt',
        'linear': './work_dir/cobot_3views_2D_xsub_medgap_aug1/linear_eval/best_model.pt',
        'finetune': './work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune/best_model.pt'
    }
    
    available_models = {}
    for model_type, path in model_files.items():
        if os.path.exists(path):
            available_models[model_type] = path
            print(f"✅ {model_type.upper()} model found: {path}")
        else:
            print(f"❌ {model_type.upper()} model not found: {path}")
    
    return available_models

def main():
    parser = argparse.ArgumentParser(description='Enhanced COBOT Visualization Analysis')
    parser.add_argument('--model_type', type=str, default='linear', 
                       choices=['pretrain', 'linear', 'finetune'],
                       help='Which model weights to use for feature extraction')
    parser.add_argument('--max_samples_per_class', type=int, default=80,
                       help='Maximum samples per class for balancing')
    parser.add_argument('--z_threshold', type=float, default=3.0,
                       help='Z-score threshold for outlier removal')
    parser.add_argument('--perplexity', type=int, default=20,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--use_umap', action='store_true',
                       help='Also compute UMAP embedding')
    
    args = parser.parse_args()
    
    print("=== Enhanced COBOT Dataset Visualization and Analysis ===")
    print("This will generate enhanced t-SNE plots and calculate NMI scores")
    print("with better clustering, outlier removal, and class balancing.")
    print()
    
    # Check available model files
    print("Checking available model files...")
    available_models = check_model_files()
    
    if not available_models:
        print("❌ No model files found! Please run training first.")
        return
    
    # Select model based on user choice
    if args.model_type in available_models:
        model_path = available_models[args.model_type]
        print(f"\n✅ Using {args.model_type.upper()} model: {model_path}")
    else:
        print(f"\n❌ {args.model_type.upper()} model not found!")
        print("Available models:", list(available_models.keys()))
        print("Using the first available model...")
        model_type = list(available_models.keys())[0]
        model_path = available_models[model_type]
        print(f"✅ Using {model_type.upper()} model: {model_path}")
    
    # Configuration
    config = {
        'model_path': model_path,
        'data_path': './cobot_med_frame64/xsub/val_position.npy',
        'label_path': './cobot_med_frame64/xsub/val_label.pkl',
        'num_classes': 19
    }
    
    # Check if required files exist
    print("\nChecking required files...")
    for key, path in config.items():
        if key != 'model_path':  # Already checked
            if os.path.exists(path):
                print(f"✅ {key}: {path}")
            else:
                print(f"❌ {key}: {path}")
                print(f"   File not found! Please check the path.")
                return
    
    print("\n" + "="*60)
    print("MODEL WEIGHTS RECOMMENDATION:")
    print("="*60)
    print("1. PRETRAIN weights: Best for understanding learned representations")
    print("   - Shows what the model learned during self-supervised training")
    print("   - Good for analyzing feature quality before classification")
    print()
    print("2. LINEAR weights: Best for analyzing final classification features")
    print("   - Shows features after training the classification head")
    print("   - Good for understanding class separability")
    print()
    print("3. FINETUNE weights: Best for analyzing fully optimized features")
    print("   - Shows features after full network optimization")
    print("   - Good for understanding final model performance")
    print()
    print("RECOMMENDATION: Use LINEAR weights for most analysis")
    print("="*60)
    print()
    
    try:
        # Initialize analyzer
        print("Initializing enhanced visualization analyzer...")
        analyzer = EnhancedCOBOTVisualizationAnalyzer(
            model_path=config['model_path'],
            data_path=config['data_path'],
            label_path=config['label_path'],
            num_classes=config['num_classes']
        )
        
        # Run enhanced analysis
        nmi_score = analyzer.run_complete_analysis(
            max_samples_per_class=args.max_samples_per_class,
            z_threshold=args.z_threshold,
            perplexity=args.perplexity,
            use_umap=args.use_umap
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 NMI Score: {nmi_score:.4f}")
        print(f"📁 Results saved in: visualization_results_enhanced/")
        print(f"🖼️  Plots generated:")
        print(f"   - Enhanced visualization (4-panel)")
        print(f"   - Individual t-SNE plot")
        if args.use_umap:
            print(f"   - Individual UMAP plot")
        print(f"   - Results summary")
        print()
        print("🎯 INTERPRETATION:")
        if nmi_score > 0.7:
            print("   Excellent clustering! Classes are well separated.")
        elif nmi_score > 0.5:
            print("   Good clustering! Classes show reasonable separation.")
        elif nmi_score > 0.3:
            print("   Fair clustering. Some class overlap exists.")
        else:
            print("   Poor clustering. Classes are not well separated.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        print("Please check the error message and try again.")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
