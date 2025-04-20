#!/usr/bin/env python3
# coding: utf-8

"""
MBSTOI Wrapper for Binaural Speech Enhancement Evaluation

This script provides a convenient wrapper around the MBSTOI implementation
to make it easier to use in the evaluation pipeline.

The MBSTOI implementation comes from the paper:
A. H. Andersen, J. M. de Haan, Z.-H. Tan, and J. Jensen, "Refinement
and validation of the binaural short time objective intelligibility
measure for spatially diverse conditions," Speech Communication,
vol. 102, pp. 1-13, Sep. 2018.
"""

import os
import sys
import numpy as np
import torch
import importlib.util

# First, check if MBSTOI module is already in the path
try:
    from MBSTOI.mbstoi import mbstoi as mbstoi_func
    MBSTOI_AVAILABLE = True
except ImportError:
    MBSTOI_AVAILABLE = False
    print("MBSTOI module not found in path. Will try to load from local files.")

# If MBSTOI wasn't found in the path, try to set it up from local files
if not MBSTOI_AVAILABLE:
    try:
        # Define required modules based on the files you provided
        modules = {
            'ec': 'ec.py',
            'mbstoi': 'mbstoi.py',
            'mbstoi_beta': 'mbstoi_beta.py',
            'remove_silent_frames': 'remove_silent_frames.py',
            'stft': 'stft.py',
            'thirdoct': 'thirdoct.py'
        }
        
        # Create a temporary MBSTOI package
        os.makedirs('MBSTOI', exist_ok=True)
        
        # Create an empty __init__.py file
        with open('MBSTOI/__init__.py', 'w') as f:
            f.write("# MBSTOI package\n")
        
        # Import necessary functions
        for module_name, file_name in modules.items():
            # Skip files that don't exist
            if not os.path.exists(file_name):
                print(f"Warning: {file_name} not found, skipping.")
                continue
                
            # Import the module from file
            spec = importlib.util.spec_from_file_location(f"MBSTOI.{module_name}", file_name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Add the module to sys.modules
            sys.modules[f"MBSTOI.{module_name}"] = module
            
            # Copy the module to the MBSTOI directory
            with open(f'MBSTOI/{module_name}.py', 'w') as f:
                with open(file_name, 'r') as original:
                    f.write(original.read())
        
        # Try to import MBSTOI again
        from MBSTOI.mbstoi import mbstoi as mbstoi_func
        MBSTOI_AVAILABLE = True
        print("Successfully set up MBSTOI from local files.")
    except Exception as e:
        print(f"Failed to set up MBSTOI: {e}")
        MBSTOI_AVAILABLE = False

def calculate_mbstoi(clean_left, clean_right, processed_left, processed_right, fs=16000):
    """
    Calculate the Modified Binaural Short-Time Objective Intelligibility (MBSTOI) measure.
    
    Args:
        clean_left (ndarray): Clean reference signal, left ear
        clean_right (ndarray): Clean reference signal, right ear
        processed_left (ndarray): Processed signal, left ear
        processed_right (ndarray): Processed signal, right ear
        fs (int): Sampling rate (default: 16000)
        
    Returns:
        float: MBSTOI score between 0 and 1
    """
    if not MBSTOI_AVAILABLE:
        print("MBSTOI calculation not available.")
        return 0
    
    try:
        # Ensure all inputs are numpy arrays
        clean_left = np.asarray(clean_left)
        clean_right = np.asarray(clean_right)
        processed_left = np.asarray(processed_left)
        processed_right = np.asarray(processed_right)
        
        # If inputs are torch tensors, convert to numpy
        if isinstance(clean_left, torch.Tensor):
            clean_left = clean_left.numpy()
        if isinstance(clean_right, torch.Tensor):
            clean_right = clean_right.numpy()
        if isinstance(processed_left, torch.Tensor):
            processed_left = processed_left.numpy()
        if isinstance(processed_right, torch.Tensor):
            processed_right = processed_right.numpy()
            
        # Ensure signals have same length
        min_length = min(len(clean_left), len(clean_right), len(processed_left), len(processed_right))
        clean_left = clean_left[:min_length]
        clean_right = clean_right[:min_length]
        processed_left = processed_left[:min_length]
        processed_right = processed_right[:min_length]
        
        # Calculate MBSTOI
        mbstoi_score = mbstoi_func(clean_left, clean_right, processed_left, processed_right)
        return mbstoi_score
    except Exception as e:
        print(f"Error calculating MBSTOI: {e}")
        return 0

if __name__ == "__main__":
    # Test the MBSTOI calculation with sample data
    import soundfile as sf
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate MBSTOI for binaural audio files")
    parser.add_argument("--clean_left", type=str, required=True, help="Path to clean left channel audio")
    parser.add_argument("--clean_right", type=str, required=True, help="Path to clean right channel audio")
    parser.add_argument("--proc_left", type=str, required=True, help="Path to processed left channel audio")
    parser.add_argument("--proc_right", type=str, required=True, help="Path to processed right channel audio")
    parser.add_argument("--fs", type=int, default=16000, help="Sampling rate")
    
    args = parser.parse_args()
    
    # Load audio files
    clean_left, fs_clean_left = sf.read(args.clean_left)
    clean_right, fs_clean_right = sf.read(args.clean_right)
    proc_left, fs_proc_left = sf.read(args.proc_left)
    proc_right, fs_proc_right = sf.read(args.proc_right)
    
    # Check sampling rates
    if fs_clean_left != args.fs or fs_clean_right != args.fs or fs_proc_left != args.fs or fs_proc_right != args.fs:
        print(f"Warning: Sampling rates don't match. Using {args.fs} as specified.")
    
    # Calculate MBSTOI
    mbstoi_score = calculate_mbstoi(clean_left, clean_right, proc_left, proc_right, args.fs)
    print(f"MBSTOI score: {mbstoi_score:.4f}")
