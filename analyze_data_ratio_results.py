#!/usr/bin/env python3
"""
Script to analyze and compare results from different data ratio finetuning experiments
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_result_files():
    """Find all result files from different data ratio experiments"""
    base_dir = "work_dir/cobot_3views_2D_xsub_medgap_aug1"
    
    experiments = {
        "100%": f"{base_dir}/finetune",
        "80%": f"{base_dir}/finetune_80percent", 
        "60%": f"{base_dir}/finetune_60percent",
        "40%": f"{base_dir}/finetune_40percent"
    }
    
    results = {}
    
    for ratio, exp_dir in experiments.items():
        if os.path.exists(exp_dir):
            # Look for log files or result files
            log_files = glob.glob(f"{exp_dir}/*.log")
            json_files = glob.glob(f"{exp_dir}/*.json")
            
            results[ratio] = {
                "dir": exp_dir,
                "log_files": log_files,
                "json_files": json_files,
                "exists": True
            }
        else:
            results[ratio] = {
                "dir": exp_dir,
                "exists": False
            }
    
    return results

def extract_accuracy_from_logs(log_file):
    """Extract accuracy values from log file"""
    accuracies = []
    epochs = []
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if "Mean_acc" in line or "test_acc" in line or "accuracy" in line.lower():
                    # Try to extract accuracy value
                    parts = line.strip().split()
                    for i, part in enumerate(parts):
                        try:
                            if "acc" in part.lower() and i+1 < len(parts):
                                acc_val = float(parts[i+1])
                                if 0 <= acc_val <= 1:
                                    accuracies.append(acc_val)
                                elif 0 <= acc_val <= 100:
                                    accuracies.append(acc_val / 100.0)
                        except:
                            continue
                            
                if "epoch" in line.lower():
                    # Try to extract epoch number
                    parts = line.strip().split()
                    for part in parts:
                        try:
                            if part.isdigit():
                                epochs.append(int(part))
                                break
                        except:
                            continue
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return accuracies, epochs

def analyze_results():
    """Analyze results from all experiments"""
    print("🔍 Analyzing Data Ratio Finetuning Results")
    print("=" * 50)
    
    results = find_result_files()
    
    # Check which experiments completed
    completed_experiments = []
    for ratio, info in results.items():
        if info["exists"]:
            print(f"✅ {ratio} data experiment: {info['dir']}")
            completed_experiments.append(ratio)
        else:
            print(f"❌ {ratio} data experiment: Not found")
    
    if not completed_experiments:
        print("❌ No completed experiments found!")
        return
    
    print(f"\n📊 Found {len(completed_experiments)} completed experiments")
    
    # Extract results from each experiment
    experiment_results = {}
    
    for ratio in completed_experiments:
        info = results[ratio]
        
        print(f"\n🔍 Analyzing {ratio} data experiment...")
        
        # Try to find the best accuracy
        best_acc = 0.0
        final_acc = 0.0
        
        # Check log files
        for log_file in info["log_files"]:
            accuracies, epochs = extract_accuracy_from_logs(log_file)
            if accuracies:
                best_acc = max(best_acc, max(accuracies))
                final_acc = accuracies[-1] if accuracies else 0.0
        
        # Check for saved model performance (look for best_model.pt)
        best_model_path = os.path.join(info["dir"], "best_model.pt")
        if os.path.exists(best_model_path):
            print(f"   ✅ Best model saved: {best_model_path}")
        
        experiment_results[ratio] = {
            "best_accuracy": best_acc,
            "final_accuracy": final_acc,
            "directory": info["dir"]
        }
        
        print(f"   📈 Best accuracy: {best_acc:.3f}")
        print(f"   📊 Final accuracy: {final_acc:.3f}")
    
    # Create summary table
    print(f"\n{'='*60}")
    print("📋 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"{'Data Ratio':<12} {'Best Acc':<12} {'Final Acc':<12} {'Status':<12}")
    print("-" * 60)
    
    for ratio in ["100%", "80%", "60%", "40%"]:
        if ratio in experiment_results:
            result = experiment_results[ratio]
            status = "✅ Complete"
            print(f"{ratio:<12} {result['best_accuracy']:<12.3f} {result['final_accuracy']:<12.3f} {status:<12}")
        else:
            print(f"{ratio:<12} {'N/A':<12} {'N/A':<12} {'❌ Missing':<12}")
    
    # Create visualization if matplotlib is available
    try:
        create_comparison_plot(experiment_results)
    except ImportError:
        print("\n📊 Matplotlib not available for plotting")
    except Exception as e:
        print(f"\n❌ Error creating plot: {e}")
    
    return experiment_results

def create_comparison_plot(experiment_results):
    """Create comparison plot of results"""
    if not experiment_results:
        return
    
    ratios = []
    best_accs = []
    final_accs = []
    
    for ratio in ["40%", "60%", "80%", "100%"]:
        if ratio in experiment_results:
            ratios.append(ratio)
            best_accs.append(experiment_results[ratio]["best_accuracy"] * 100)  # Convert to percentage
            final_accs.append(experiment_results[ratio]["final_accuracy"] * 100)
    
    if len(ratios) < 2:
        print("📊 Need at least 2 experiments for comparison plot")
        return
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    x = np.arange(len(ratios))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, best_accs, width, label='Best Accuracy', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, final_accs, width, label='Final Accuracy', alpha=0.8, color='lightcoral')
    
    ax.set_xlabel('Training Data Ratio')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('COBOT Finetuning: Performance vs Training Data Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(ratios)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "data_ratio_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Comparison plot saved: {plot_path}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

def main():
    """Main function"""
    print("🔬 COBOT Data Ratio Analysis")
    print("=" * 40)
    
    experiment_results = analyze_results()
    
    print(f"\n💡 Analysis Tips:")
    print(f"   • Compare how performance changes with less training data")
    print(f"   • Look for the point of diminishing returns")
    print(f"   • Consider computational cost vs performance trade-off")
    print(f"   • Use results to determine minimum viable training set size")
    
    print(f"\n📁 Individual experiment results can be found in:")
    for ratio in ["100%", "80%", "60%", "40%"]:
        exp_dir = f"work_dir/cobot_3views_2D_xsub_medgap_aug1/finetune"
        if ratio != "100%":
            exp_dir += f"_{ratio.replace('%', 'percent')}"
        print(f"   • {ratio} data: {exp_dir}")

if __name__ == "__main__":
    main()
