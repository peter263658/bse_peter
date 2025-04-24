#!/usr/bin/env python
# coding: utf-8

"""
Script to evaluate BCCTN model on specific SNR levels
This script runs the evaluation on each SNR level directory separately
and combines the results
"""

import os
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_snr_directories(base_dir):
    """Find all SNR-specific directories in the dataset directory"""
    snr_dirs = {}
    
    # Print the directory structure for debugging
    print(f"Checking directory structure in: {base_dir}")
    if os.path.exists(base_dir):
        print("Directory exists, contents:")
        for item in os.listdir(base_dir):
            print(f"  - {item}")
    else:
        print(f"WARNING: Directory {base_dir} does not exist!")
        return {}
    
    # First try direct SNR subdirectories in the format "SNR_XdB"
    for item in os.listdir(base_dir):
        if item.startswith("SNR_") and item.endswith("dB"):
            try:
                # Extract SNR level, e.g., "SNR_-6dB" -> -6
                snr_level = int(item.replace("SNR_", "").replace("dB", ""))
                snr_dirs[snr_level] = os.path.join(base_dir, item)
                print(f"Found SNR directory: {item} -> {snr_level}")
            except ValueError:
                print(f"Warning: Could not parse SNR level from {item}")
    
    # If we found SNR directories, return them
    if snr_dirs:
        return snr_dirs
    
    # If we didn't find direct SNR directories, check if the base_dir itself 
    # is named as an SNR directory
    base_name = os.path.basename(base_dir)
    if base_name.startswith("SNR_") and base_name.endswith("dB"):
        try:
            snr_level = int(base_name.replace("SNR_", "").replace("dB", ""))
            snr_dirs[snr_level] = base_dir
            print(f"Base directory is an SNR directory: {base_name} -> {snr_level}")
        except ValueError:
            print(f"Warning: Could not parse SNR level from base directory {base_name}")
    
    return snr_dirs

def evaluate_on_snr_level(snr_level, clean_dir, noisy_dir, checkpoint_path, output_base_dir, config_path, num_samples):
    """Run evaluation on a specific SNR level"""
    print(f"\n\n{'='*50}")
    print(f"Evaluating on SNR level: {snr_level} dB")
    print(f"Clean directory: {clean_dir}")
    print(f"Noisy directory: {noisy_dir}")
    print(f"{'='*50}\n")
    
    output_dir = os.path.join(output_base_dir, f"SNR_{snr_level}dB")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the evaluation
    cmd = [
        "python", "evaluate_model.py",  # Use the actual script name
        "--config_path", config_path,
        "--checkpoint", checkpoint_path,
        "--noisy_data", str(noisy_dir),
        "--clean_data", str(clean_dir),
        "--output_dir", output_dir,
        "--num_samples", str(num_samples),
        "--by_snr"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)
    
    return output_dir

def combine_results(output_dirs, output_base_dir):
    """Combine results from multiple SNR levels into a single report"""
    # Collect all CSV results
    metrics_by_snr = {}
    
    for snr_level, output_dir in output_dirs.items():
        # Load metrics CSV
        csv_path = os.path.join(output_dir, "overall_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            metrics_dict = dict(zip(df.Metric, df.Value))
            metrics_by_snr[snr_level] = metrics_dict
    
    # Create a summary report
    summary_file = os.path.join(output_base_dir, "summary_report.md")
    with open(summary_file, 'w') as f:
        f.write("# BCCTN Evaluation Results by SNR Level\n\n")
        
        # SegSNR Table
        f.write("## SegSNR Improvement (dB)\n\n")
        f.write("| SNR (dB) | Noisy (L) | Enhanced (L) | Improvement (L) | Noisy (R) | Enhanced (R) | Improvement (R) |\n")
        f.write("|----------|-----------|--------------|-----------------|-----------|--------------|------------------|\n")
        
        sorted_snr_levels = sorted(metrics_by_snr.keys())
        for snr_level in sorted_snr_levels:
            metrics = metrics_by_snr[snr_level]
            noisy_l = metrics.get('SegSNR Noisy (L)', 0)
            enhanced_l = metrics.get('SegSNR Enhanced (L)', 0)
            improvement_l = metrics.get('SegSNR Improvement (L)', 0)
            noisy_r = metrics.get('SegSNR Noisy (R)', 0)
            enhanced_r = metrics.get('SegSNR Enhanced (R)', 0)
            improvement_r = metrics.get('SegSNR Improvement (R)', 0)
            
            f.write(f"| {snr_level} | {noisy_l:.2f} | {enhanced_l:.2f} | **{improvement_l:.2f}** | {noisy_r:.2f} | {enhanced_r:.2f} | **{improvement_r:.2f}** |\n")
        
        # MBSTOI Table
        f.write("\n## MBSTOI Improvement\n\n")
        f.write("| SNR (dB) | Noisy | Enhanced | Improvement |\n")
        f.write("|----------|-------|----------|-------------|\n")
        
        for snr_level in sorted_snr_levels:
            metrics = metrics_by_snr[snr_level]
            noisy = metrics.get('MBSTOI Noisy', 0)
            enhanced = metrics.get('MBSTOI Enhanced', 0)
            improvement = metrics.get('MBSTOI Improvement', 0)
            
            f.write(f"| {snr_level} | {noisy:.3f} | {enhanced:.3f} | **{improvement:.3f}** |\n")
        
        # STOI Table
        f.write("\n## STOI Improvement\n\n")
        f.write("| SNR (dB) | Noisy (L) | Enhanced (L) | Improvement (L) | Noisy (R) | Enhanced (R) | Improvement (R) |\n")
        f.write("|----------|-----------|--------------|-----------------|-----------|--------------|------------------|\n")
        
        for snr_level in sorted_snr_levels:
            metrics = metrics_by_snr[snr_level]
            noisy_l = metrics.get('STOI Noisy (L)', 0)
            enhanced_l = metrics.get('STOI Enhanced (L)', 0)
            improvement_l = metrics.get('STOI Improvement (L)', 0)
            noisy_r = metrics.get('STOI Noisy (R)', 0)
            enhanced_r = metrics.get('STOI Enhanced (R)', 0)
            improvement_r = metrics.get('STOI Improvement (R)', 0)
            
            f.write(f"| {snr_level} | {noisy_l:.3f} | {enhanced_l:.3f} | **{improvement_l:.3f}** | {noisy_r:.3f} | {enhanced_r:.3f} | **{improvement_r:.3f}** |\n")
        
        # Interaural Cues Table
        f.write("\n## Interaural Cue Preservation\n\n")
        f.write("| SNR (dB) | ILD Error (dB) | IPD Error (°) |\n")
        f.write("|----------|----------------|---------------|\n")
        
        for snr_level in sorted_snr_levels:
            metrics = metrics_by_snr[snr_level]
            ild_error = metrics.get('ILD Error', 0)
            ipd_error = metrics.get('IPD Error', 0)
            
            f.write(f"| {snr_level} | {ild_error:.2f} | {ipd_error:.2f} |\n")
    
    # Generate plots
    create_summary_plots(metrics_by_snr, output_base_dir)
    
    print(f"Summary report saved to {summary_file}")

def create_summary_plots(metrics_by_snr, output_dir):
    """Create summary plots for all metrics across SNR levels"""
    
    sorted_snr_levels = sorted(metrics_by_snr.keys())
    
    # SegSNR improvement plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('SegSNR Improvement (L)', 0) for snr in sorted_snr_levels],
             'o-', label='Left Channel')
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('SegSNR Improvement (R)', 0) for snr in sorted_snr_levels],
             'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('SegSNR Improvement (dB)')
    plt.title('SegSNR Improvement by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "segsnr_improvement_summary.png"))
    plt.close()
    
    # MBSTOI improvement plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('MBSTOI Improvement', 0) for snr in sorted_snr_levels],
             'o-')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('MBSTOI Improvement')
    plt.title('MBSTOI Improvement by SNR Level')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mbstoi_improvement_summary.png"))
    plt.close()
    
    # STOI improvement plot
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('STOI Improvement (L)', 0) for snr in sorted_snr_levels],
             'o-', label='Left Channel')
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('STOI Improvement (R)', 0) for snr in sorted_snr_levels],
             'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('STOI Improvement')
    plt.title('STOI Improvement by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "stoi_improvement_summary.png"))
    plt.close()
    
    # Interaural cue error plots
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('ILD Error', 0) for snr in sorted_snr_levels],
             'o-', label='ILD Error (dB)')
    plt.plot(sorted_snr_levels, 
             [metrics_by_snr[snr].get('IPD Error', 0) / 10 for snr in sorted_snr_levels],
             'o-', label='IPD Error (°/10)')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('Error')
    plt.title('Interaural Cue Preservation by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "interaural_errors_summary.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate BCCTN model on specific SNR levels")
    parser.add_argument("--noisy_base", type=str, required=True, help="Base directory containing noisy test data with SNR subfolders")
    parser.add_argument("--clean_base", type=str, required=True, help="Base directory containing clean test data with SNR subfolders")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="snr_evaluation_results", help="Directory to save results")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to config directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate per SNR level")
    parser.add_argument("--snr_levels", type=int, nargs="+", default=None, help="Specific SNR levels to evaluate (e.g., -6 -3 0 3)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find SNR directories
    noisy_snr_dirs = find_snr_directories(args.noisy_base)
    clean_snr_dirs = find_snr_directories(args.clean_base)
    
    print(f"Found noisy SNR directories: {list(noisy_snr_dirs.keys())}")
    print(f"Found clean SNR directories: {list(clean_snr_dirs.keys())}")
    
    # If no SNR directories found, the supplied paths might be the direct SNR directories
    # Try to extract SNR level from the base directories themselves
    if not noisy_snr_dirs and not clean_snr_dirs:
        # Try analyzing the directory structure
        print("No SNR directories found in the provided paths.")
        print("Checking if the paths themselves contain SNR-level data...")
        
        # Path structure might be like:
        # Dataset/
        # ├── clean_testset_1f_SNR_-6dB/
        # ├── clean_testset_1f_SNR_-3dB/
        # ├── ...
        # ├── noisy_testset_1f_SNR_-6dB/
        # ├── noisy_testset_1f_SNR_-3dB/
        # ├── ...
        
        # Try to get SNR level from path components
        def extract_snr_from_path(path):
            for component in str(path).split('/'):
                if "SNR_" in component:
                    try:
                        snr_part = component.split("SNR_")[1].split("dB")[0]
                        return int(snr_part)
                    except:
                        return None
            return None
            
        noisy_snr = extract_snr_from_path(args.noisy_base)
        clean_snr = extract_snr_from_path(args.clean_base)
        
        if noisy_snr is not None and clean_snr is not None and noisy_snr == clean_snr:
            print(f"Found matching SNR level in paths: {noisy_snr}")
            noisy_snr_dirs = {noisy_snr: args.noisy_base}
            clean_snr_dirs = {clean_snr: args.clean_base}
    
    # Filter by specified SNR levels if provided
    if args.snr_levels:
        noisy_snr_dirs = {k: v for k, v in noisy_snr_dirs.items() if k in args.snr_levels}
        clean_snr_dirs = {k: v for k, v in clean_snr_dirs.items() if k in args.snr_levels}
        print(f"Filtered to SNR levels: {args.snr_levels}")
    
    # Make sure we have matching clean and noisy directories
    snr_levels = sorted(set(noisy_snr_dirs.keys()) & set(clean_snr_dirs.keys()))
    if not snr_levels:
        print("Error: No matching SNR levels found in clean and noisy directories")
        return
    
    print(f"Will evaluate on SNR levels: {snr_levels}")
    
    # Run evaluation for each SNR level
    output_dirs = {}
    for snr_level in snr_levels:
        noisy_dir = noisy_snr_dirs[snr_level]
        clean_dir = clean_snr_dirs[snr_level]
        
        output_dir = evaluate_on_snr_level(
            snr_level, 
            clean_dir, 
            noisy_dir, 
            args.checkpoint, 
            args.output_dir, 
            args.config_path, 
            args.num_samples
        )
        
        output_dirs[snr_level] = output_dir
    
    # Combine results
    combine_results(output_dirs, args.output_dir)
    
if __name__ == "__main__":
    main()