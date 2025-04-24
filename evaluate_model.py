#!/usr/bin/env python
# coding: utf-8

"""
Simplified Binaural Speech Enhancement Model Evaluator

This script evaluates a trained BCCTN model on binaural speech signals with different SNR levels.
It produces comprehensive metrics for each SNR level including:
- STOI (Short-Time Objective Intelligibility)
- MBSTOI (Modified Binaural STOI)
- SegSNR (Segmental Signal-to-Noise Ratio)
- ILD and IPD preservation (Interaural Level and Phase Differences)

Usage:
  python evaluate_model.py --checkpoint MODEL_CHECKPOINT 
                          --data_root DATA_DIR 
                          [--output_dir OUTPUT_DIR]
                          [--num_samples N]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
import re
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

# Import model-specific components
from DCNN.trainer import DCNNLightningModule
from DCNN.feature_extractors import Stft, IStft
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

# Try to import MBSTOI
try:
    from MBSTOI.mbstoi import mbstoi
    MBSTOI_AVAILABLE = True
    print("MBSTOI library available")
except ImportError:
    try:
        # Alternative import path
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from MBSTOI.mbstoi import mbstoi
        MBSTOI_AVAILABLE = True
        print("MBSTOI library available (via alternative path)")
    except ImportError:
        print("Warning: MBSTOI not available. Install the MBSTOI package for binaural intelligibility metrics.")
        MBSTOI_AVAILABLE = False

# Define constants
SR = 16000  # Sample rate
FFT_LEN = 512
WIN_LEN = 400
WIN_INC = 100

class Config:
    fs = SR  # Default sampling rate

CONFIG = Config()

def prepare_for_stft(x):
    """Convert 1D array to 3D tensor with shape (batch=1, channel=1, time)"""
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

def extract_snr_from_filename(filename):
    """Extract SNR value from filename"""
    # Pattern matching for SNR in filename
    pattern = r'_snr([+-]\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return None

def find_files_by_snr(data_dir, snr_level):
    """Find all .wav files with a specific SNR level in the directory"""
    files = []
    
    # Check if there's an SNR-specific subdirectory
    snr_dir = os.path.join(data_dir, f"SNR_{snr_level}dB")
    if os.path.exists(snr_dir):
        # If SNR directory exists, use all files in it
        for file in Path(snr_dir).glob("*.wav"):
            files.append(str(file))
        print(f"Found {len(files)} files in SNR directory: {snr_dir}")
        return files
    
    # If no SNR directory, search by filename pattern
    pattern = f"_snr{snr_level:+d}" if snr_level >= 0 else f"_snr{snr_level}"
    for file in Path(data_dir).rglob("*.wav"):
        if pattern in file.name:
            files.append(str(file))
    
    print(f"Found {len(files)} files for SNR level {snr_level} dB by filename pattern")
    return files

def calculate_mbstoi(clean_l, clean_r, proc_l, proc_r):
    """Calculate MBSTOI with error handling"""
    if not MBSTOI_AVAILABLE:
        return 0
    
    try:
        # Ensure arrays are numpy arrays with correct shape
        clean_l = np.asarray(clean_l).flatten()
        clean_r = np.asarray(clean_r).flatten()
        proc_l = np.asarray(proc_l).flatten()
        proc_r = np.asarray(proc_r).flatten()
        
        # Trim to same length if needed
        min_len = min(len(clean_l), len(clean_r), len(proc_l), len(proc_r))
        clean_l = clean_l[:min_len]
        clean_r = clean_r[:min_len]
        proc_l = proc_l[:min_len]
        proc_r = proc_r[:min_len]
        
        # Calculate MBSTOI
        result = mbstoi(clean_l, clean_r, proc_l, proc_r)
        return result
    except Exception as e:
        print(f"Error calculating MBSTOI: {e}")
        return 0

def align_signals(clean, processed, max_delay=64):
    """Align signals using cross-correlation to compensate for processing delay"""
    # Ensure same length for correlation
    min_len = min(len(clean), len(processed))
    clean_temp = clean[:min_len].copy()
    processed_temp = processed[:min_len].copy()
    
    # Compute cross-correlation
    corr = np.correlate(clean_temp, processed_temp, mode='full')
    max_idx = np.argmax(np.abs(corr))
    center = len(corr) // 2
    delay = max_idx - center
    
    # Limit to max_delay
    delay = max(min(delay, max_delay), -max_delay)
    
    # Apply delay
    if delay > 0:
        # processed is delayed
        clean_aligned = clean_temp[delay:]
        processed_aligned = processed_temp[:-delay] if delay > 0 else processed_temp
    else:
        # clean is delayed
        clean_aligned = clean_temp[:delay] if delay < 0 else clean_temp
        processed_aligned = processed_temp[-delay:]
        
    # Ensure same length after alignment
    min_len = min(len(clean_aligned), len(processed_aligned))
    clean_aligned = clean_aligned[:min_len]
    processed_aligned = processed_aligned[:min_len]
    
    return clean_aligned, processed_aligned

def calculate_fw_seg_snr(clean, processed, frame_size=256, hop=128, fs=16000):
    """
    Calculate frequency-weighted Segmental SNR
    """
    # Align signals
    clean_aligned, processed_aligned = align_signals(clean, processed)
    
    # Split into frames
    num_frames = (len(clean_aligned) - frame_size) // hop + 1
    
    if num_frames <= 0:
        return -10  # Return a low value for very short signals
    
    # Initialize arrays for snr values
    seg_snr_values = np.zeros(num_frames)
    
    for i in range(num_frames):
        start = i * hop
        clean_frame = clean_aligned[start:start+frame_size]
        proc_frame = processed_aligned[start:start+frame_size]
        
        # Calculate noise frame
        noise_frame = proc_frame - clean_frame
        
        # Compute FFT
        clean_fft = np.fft.rfft(clean_frame * np.hanning(frame_size))
        noise_fft = np.fft.rfft(noise_frame * np.hanning(frame_size))
        
        # Compute power spectrum
        clean_power = np.abs(clean_fft)**2
        noise_power = np.abs(noise_fft)**2 + 1e-10  # Avoid division by zero
        
        # Calculate critical band frequencies
        num_crit_bands = 25
        freq_bands = librosa.filters.mel(sr=fs, n_fft=frame_size, n_mels=num_crit_bands)
        
        # Apply frequency weighting
        clean_power_bands = np.dot(freq_bands, clean_power)
        noise_power_bands = np.dot(freq_bands, noise_power)
        
        # Compute SNR for each band
        band_snr = 10 * np.log10((clean_power_bands + 1e-10) / (noise_power_bands + 1e-10))
        
        # Limit SNR range to [−10, 35] dB as recommended
        band_snr = np.clip(band_snr, -10, 35)
        
        # Average across bands
        seg_snr_values[i] = np.mean(band_snr)
    
    # Return the mean seg SNR
    return np.mean(seg_snr_values)

def compute_ild_db(s1, s2, eps=1e-6):
    """Compute Interaural Level Difference in dB"""
    l1 = 20 * np.log10(np.abs(s1) + eps)
    l2 = 20 * np.log10(np.abs(s2) + eps)
    return l1 - l2

def compute_ipd_rad(s1, s2, eps=1e-6):
    """Compute Interaural Phase Difference in radians"""
    # Calculate phase difference and wrap to [-π, π]
    phase_diff = np.angle(s1 * np.conj(s2) + eps)
    return phase_diff

def speech_mask(stft_l, stft_r, threshold=20):
    """Create a speech binary mask using the energy level"""
    # Calculate energy (in dB) from both channels
    energy = 10 * np.log10(np.abs(stft_l)**2 + np.abs(stft_r)**2 + 1e-12)
    
    # Compute maximum energy per frame (time)
    max_per_frame = np.max(energy, axis=0, keepdims=True)
    
    # Create mask: select bins above threshold relative to max
    mask = (energy >= max_per_frame - threshold)
    
    # Ensure we have enough active bins (minimum percentage)
    active_bins = np.sum(mask)
    total_bins = mask.size
    active_ratio = active_bins / total_bins
    
    if active_ratio < 0.1:  # If less than 10% active, adjust threshold
        # Sort energies and take top 10%
        flattened = energy.flatten()
        sorted_energy = np.sort(flattened)[::-1]  # Sort descending
        new_threshold = sorted_energy[int(0.1 * total_bins)]
        mask = (energy >= new_threshold)
    
    return mask

def calculate_ild_loss(target_stft_l, target_stft_r, output_stft_l, output_stft_r):
    """Calculate ILD loss between target and output signals"""
    # Calculate ILDs
    target_ild = compute_ild_db(target_stft_l, target_stft_r)
    output_ild = compute_ild_db(output_stft_l, output_stft_r)
    
    # Create a speech activity mask
    mask = speech_mask(target_stft_l, target_stft_r, threshold=20)
    
    # Calculate the absolute difference and apply mask
    ild_error = np.abs(target_ild - output_ild)
    masked_ild_error = ild_error * mask
    
    # Average over speech-active regions
    if np.sum(mask) > 0:
        return np.sum(masked_ild_error) / np.sum(mask)
    else:
        return np.mean(ild_error)  # Fallback to all TF bins

def calculate_ipd_loss(target_stft_l, target_stft_r, output_stft_l, output_stft_r):
    """Calculate IPD loss between target and output signals in degrees"""
    # Calculate IPDs
    target_ipd = compute_ipd_rad(target_stft_l, target_stft_r)
    output_ipd = compute_ipd_rad(output_stft_l, output_stft_r)
    
    # Create a speech activity mask
    mask = speech_mask(target_stft_l, target_stft_r)
    
    # Calculate the absolute difference and apply mask
    # Convert to degrees for more intuitive interpretation
    ipd_error = np.abs(target_ipd - output_ipd) * (180 / np.pi)
    masked_ipd_error = ipd_error * mask
    
    # Average over speech-active regions
    if np.sum(mask) > 0:
        return np.sum(masked_ipd_error) / np.sum(mask)
    else:
        return 0

def evaluate_file_pair(model, clean_path, noisy_path, device):
    """Evaluate a single pair of clean and noisy files"""
    try:
        # Load audio files
        clean, sr_clean = sf.read(clean_path)
        noisy, sr_noisy = sf.read(noisy_path)
        
        # Ensure correct shape (should be [samples, 2] for stereo)
        if clean.ndim == 1:
            print(f"Warning: {clean_path} is mono, expected stereo")
            return None
        if noisy.ndim == 1:
            print(f"Warning: {noisy_path} is mono, expected stereo")
            return None
        
        # Transpose if needed
        if clean.shape[1] != 2:
            clean = clean.T
        if noisy.shape[1] != 2:
            noisy = noisy.T
        
        # Convert to correct format for model
        noisy_tensor = torch.from_numpy(noisy.T).float().unsqueeze(0).to(device)
        
        # Process with model
        with torch.no_grad():
            enhanced_tensor = model(noisy_tensor).cpu()
        
        # Convert back to numpy
        enhanced = enhanced_tensor[0].numpy().T
        
        # Extract channels
        clean_l, clean_r = clean[:, 0], clean[:, 1]
        noisy_l, noisy_r = noisy[:, 0], noisy[:, 1]
        enhanced_l, enhanced_r = enhanced[:, 0], enhanced[:, 1]
        
        # Initialize audio processing tools
        stft = Stft(n_dft=FFT_LEN, hop_size=WIN_INC, win_length=WIN_LEN)
        stoi_metric = ShortTimeObjectiveIntelligibility(fs=SR)
        
        # Align signals for fair comparison
        clean_l_n, noisy_l = align_signals(clean_l, noisy_l)
        clean_r_n, noisy_r = align_signals(clean_r, noisy_r)
        clean_l_e, enhanced_l = align_signals(clean_l, enhanced_l)
        clean_r_e, enhanced_r = align_signals(clean_r, enhanced_r)
        
        # Calculate metrics
        # 1. SegSNR
        snr_noisy_l = calculate_fw_seg_snr(clean_l_n, noisy_l, fs=SR)
        snr_noisy_r = calculate_fw_seg_snr(clean_r_n, noisy_r, fs=SR)
        snr_enhanced_l = calculate_fw_seg_snr(clean_l_e, enhanced_l, fs=SR)
        snr_enhanced_r = calculate_fw_seg_snr(clean_r_e, enhanced_r, fs=SR)
        
        # 2. STOI
        stoi_noisy_l = stoi_metric(torch.from_numpy(clean_l_n), torch.from_numpy(noisy_l)).item()
        stoi_noisy_r = stoi_metric(torch.from_numpy(clean_r_n), torch.from_numpy(noisy_r)).item()
        stoi_enhanced_l = stoi_metric(torch.from_numpy(clean_l_e), torch.from_numpy(enhanced_l)).item()
        stoi_enhanced_r = stoi_metric(torch.from_numpy(clean_r_e), torch.from_numpy(enhanced_r)).item()
        
        # 3. MBSTOI
        mbstoi_noisy = calculate_mbstoi(clean_l_n, clean_r_n, noisy_l, noisy_r) if MBSTOI_AVAILABLE else 0
        mbstoi_enhanced = calculate_mbstoi(clean_l_e, clean_r_e, enhanced_l, enhanced_r) if MBSTOI_AVAILABLE else 0
        
        # Apply STFT for interaural cues
        noisy_stft_l = stft(prepare_for_stft(noisy_l)).squeeze(0).numpy()
        noisy_stft_r = stft(prepare_for_stft(noisy_r)).squeeze(0).numpy()
        enhanced_stft_l = stft(prepare_for_stft(enhanced_l)).squeeze(0).numpy()
        enhanced_stft_r = stft(prepare_for_stft(enhanced_r)).squeeze(0).numpy()
        clean_stft_l = stft(prepare_for_stft(clean_l_n)).squeeze(0).numpy()
        clean_stft_r = stft(prepare_for_stft(clean_r_n)).squeeze(0).numpy()
        
        # 4. Calculate ILD and IPD errors
        ild_error = calculate_ild_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
        ipd_error = calculate_ipd_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
        
        # Return all metrics
        return {
            'stoi_noisy_l': stoi_noisy_l, 
            'stoi_noisy_r': stoi_noisy_r,
            'stoi_enhanced_l': stoi_enhanced_l, 
            'stoi_enhanced_r': stoi_enhanced_r,
            'mbstoi_noisy': mbstoi_noisy,
            'mbstoi_enhanced': mbstoi_enhanced,
            'snr_noisy_l': snr_noisy_l, 
            'snr_noisy_r': snr_noisy_r,
            'snr_enhanced_l': snr_enhanced_l, 
            'snr_enhanced_r': snr_enhanced_r,
            'ild_error': ild_error,
            'ipd_error': ipd_error,
            # Add file paths for reference
            'clean_path': clean_path,
            'noisy_path': noisy_path,
            # Add audio for optional saving
            'clean_audio': clean,
            'noisy_audio': noisy,
            'enhanced_audio': enhanced
        }
    except Exception as e:
        print(f"Error processing file pair {clean_path} and {noisy_path}: {e}")
        return None

def find_matching_clean_file(noisy_path, clean_dir):
    """Find the matching clean file for a noisy file"""
    noisy_name = os.path.basename(noisy_path)
    
    # Extract SNR level from filename to look in the right SNR directory
    snr_level = extract_snr_from_filename(noisy_name)
    
    # Check if clean directory has SNR-specific subdirectories
    snr_specific_dir = None
    if snr_level is not None:
        possible_snr_dirs = [
            os.path.join(clean_dir, f"SNR_{snr_level}dB"),
            os.path.join(clean_dir, f"SNR_{snr_level:+d}dB")
        ]
        for dir_path in possible_snr_dirs:
            if os.path.exists(dir_path):
                snr_specific_dir = dir_path
                break
    
    # Search paths in order of likelihood
    search_paths = [clean_dir]  # Default: root clean directory
    if snr_specific_dir:
        search_paths.insert(0, snr_specific_dir)  # First try SNR-specific directory
    
    # Try exact filename match in each search path
    for search_path in search_paths:
        clean_path = os.path.join(search_path, noisy_name)
        if os.path.exists(clean_path):
            return clean_path
        
        # Try common variations of the name
        clean_name = noisy_name.replace("noisy_", "")
        variations = [
            clean_name,
            "clean_" + clean_name,
            noisy_name.replace("noisy", "clean")
        ]
        
        for name in variations:
            path = os.path.join(search_path, name)
            if os.path.exists(path):
                return path
    
    # If no exact match, search for files with similar identifiers
    # Extract unique identifiers from noisy filename (like speaker, utterance IDs)
    if snr_level is not None:
        parts = noisy_name.split('_')
        # Get the base pattern (usually speaker and utterance IDs)
        base_pattern = parts[0]
        if len(parts) > 1:
            base_pattern += "_" + parts[1]
            
        # Look in SNR-specific directory first if it exists
        if snr_specific_dir:
            for file in os.listdir(snr_specific_dir):
                if base_pattern in file:
                    return os.path.join(snr_specific_dir, file)
        
        # Try the main clean directory if SNR-specific search failed
        for file in os.listdir(clean_dir):
            if base_pattern in file:
                return os.path.join(clean_dir, file)
    
    print(f"Warning: No matching clean file found for {noisy_path}")
    return None

def evaluate_snr_level(model, clean_dir, noisy_dir, snr_level, device, num_samples=50, save_audio=False, output_dir=None):
    """Evaluate model performance on a specific SNR level"""
    print(f"\n{'='*50}")
    print(f"Evaluating at SNR level: {snr_level} dB")
    print(f"{'='*50}")
    
    # Find files matching the SNR level
    noisy_files = find_files_by_snr(noisy_dir, snr_level)
    
    if not noisy_files:
        print(f"No files found for SNR level {snr_level} dB. Skipping.")
        return None
    
    # If we have more files than needed, randomly sample
    if len(noisy_files) > num_samples:
        np.random.shuffle(noisy_files)
        noisy_files = noisy_files[:num_samples]
    
    # Results for this SNR level
    results = []
    
    # Process each file
    for i, noisy_path in enumerate(tqdm(noisy_files, desc=f"Processing SNR {snr_level} dB")):
        # Find matching clean file
        clean_path = find_matching_clean_file(noisy_path, clean_dir)
        
        if not clean_path:
            continue
        
        # Evaluate the pair
        metrics = evaluate_file_pair(model, clean_path, noisy_path, device)
        
        if metrics:
            results.append(metrics)
            
            # Save a few audio samples if requested
            if save_audio and output_dir and i < 5:  # Save first 5 samples
                sample_dir = os.path.join(output_dir, f"snr_{snr_level}dB_sample_{i}")
                os.makedirs(sample_dir, exist_ok=True)
                
                # Save audio files
                sf.write(os.path.join(sample_dir, "clean_L.wav"), metrics['clean_audio'][:, 0], SR)
                sf.write(os.path.join(sample_dir, "clean_R.wav"), metrics['clean_audio'][:, 1], SR)
                sf.write(os.path.join(sample_dir, "noisy_L.wav"), metrics['noisy_audio'][:, 0], SR)
                sf.write(os.path.join(sample_dir, "noisy_R.wav"), metrics['noisy_audio'][:, 1], SR)
                sf.write(os.path.join(sample_dir, "enhanced_L.wav"), metrics['enhanced_audio'][:, 0], SR)
                sf.write(os.path.join(sample_dir, "enhanced_R.wav"), metrics['enhanced_audio'][:, 1], SR)
    
    if not results:
        print(f"No valid results for SNR level {snr_level} dB. Skipping.")
        return None
    
    # Calculate average metrics
    avg_metrics = {}
    for key in results[0].keys():
        if key not in ['clean_path', 'noisy_path', 'clean_audio', 'noisy_audio', 'enhanced_audio']:
            avg_metrics[key] = np.mean([r[key] for r in results])
    
    # Calculate derived metrics
    avg_metrics['stoi_improvement_l'] = avg_metrics['stoi_enhanced_l'] - avg_metrics['stoi_noisy_l']
    avg_metrics['stoi_improvement_r'] = avg_metrics['stoi_enhanced_r'] - avg_metrics['stoi_noisy_r']
    avg_metrics['mbstoi_improvement'] = avg_metrics['mbstoi_enhanced'] - avg_metrics['mbstoi_noisy']
    avg_metrics['snr_improvement_l'] = avg_metrics['snr_enhanced_l'] - avg_metrics['snr_noisy_l']
    avg_metrics['snr_improvement_r'] = avg_metrics['snr_enhanced_r'] - avg_metrics['snr_noisy_r']
    
    # Print metrics
    print(f"\nResults for SNR level {snr_level} dB (average over {len(results)} files):")
    print(f"STOI Improvement (L/R): {avg_metrics['stoi_improvement_l']:.3f}/{avg_metrics['stoi_improvement_r']:.3f}")
    if MBSTOI_AVAILABLE:
        print(f"MBSTOI Improvement: {avg_metrics['mbstoi_improvement']:.3f}")
    print(f"SegSNR Improvement (L/R): {avg_metrics['snr_improvement_l']:.2f}/{avg_metrics['snr_improvement_r']:.2f} dB")
    print(f"ILD Error: {avg_metrics['ild_error']:.2f} dB")
    print(f"IPD Error: {avg_metrics['ipd_error']:.2f} degrees")
    
    # Save detailed metrics if output directory is provided
    if output_dir:
        snr_dir = os.path.join(output_dir, f"snr_{snr_level}dB")
        os.makedirs(snr_dir, exist_ok=True)
        
        # Save summary metrics to CSV
        metrics_df = pd.DataFrame({
            'Metric': [
                'STOI Noisy (L)', 'STOI Enhanced (L)', 'STOI Improvement (L)',
                'STOI Noisy (R)', 'STOI Enhanced (R)', 'STOI Improvement (R)',
                'MBSTOI Noisy', 'MBSTOI Enhanced', 'MBSTOI Improvement',
                'SegSNR Noisy (L)', 'SegSNR Enhanced (L)', 'SegSNR Improvement (L)',
                'SegSNR Noisy (R)', 'SegSNR Enhanced (R)', 'SegSNR Improvement (R)',
                'ILD Error', 'IPD Error'
            ],
            'Value': [
                avg_metrics['stoi_noisy_l'], avg_metrics['stoi_enhanced_l'], avg_metrics['stoi_improvement_l'],
                avg_metrics['stoi_noisy_r'], avg_metrics['stoi_enhanced_r'], avg_metrics['stoi_improvement_r'],
                avg_metrics['mbstoi_noisy'], avg_metrics['mbstoi_enhanced'], avg_metrics['mbstoi_improvement'],
                avg_metrics['snr_noisy_l'], avg_metrics['snr_enhanced_l'], avg_metrics['snr_improvement_l'],
                avg_metrics['snr_noisy_r'], avg_metrics['snr_enhanced_r'], avg_metrics['snr_improvement_r'],
                avg_metrics['ild_error'], avg_metrics['ipd_error']
            ]
        })
        metrics_df.to_csv(os.path.join(snr_dir, "metrics.csv"), index=False)
        
        # Save raw results for further analysis
        raw_data = [{k: v for k, v in r.items() if k not in ['clean_audio', 'noisy_audio', 'enhanced_audio']} 
                    for r in results]
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_csv(os.path.join(snr_dir, "raw_results.csv"), index=False)
    
    return {
        'snr_level': snr_level,
        'metrics': avg_metrics,
        'num_samples': len(results)
    }

def plot_metrics_by_snr(all_results, output_dir):
    """Create visualizations for metrics across SNR levels"""
    if not all_results:
        print("No results to plot")
        return
    
    # Extract data for plotting
    snr_levels = [r['snr_level'] for r in all_results]
    indices = np.argsort(snr_levels)
    snr_levels = [snr_levels[i] for i in indices]
    
    # Extract metrics for each dimension
    get_metric = lambda key: [all_results[i]['metrics'][key] for i in indices]
    
    # STOI Improvement
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, get_metric('stoi_improvement_l'), 'o-', label='Left Channel')
    plt.plot(snr_levels, get_metric('stoi_improvement_r'), 'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('STOI Improvement')
    plt.title('STOI Improvement by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "stoi_improvement.png"))
    plt.close()
    
    # MBSTOI Improvement
    if MBSTOI_AVAILABLE:
        plt.figure(figsize=(10, 6))
        plt.plot(snr_levels, get_metric('mbstoi_improvement'), 'o-')
        plt.xlabel('SNR Level (dB)')
        plt.ylabel('MBSTOI Improvement')
        plt.title('MBSTOI Improvement by SNR Level')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "mbstoi_improvement.png"))
        plt.close()
    
    # SegSNR Improvement
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, get_metric('snr_improvement_l'), 'o-', label='Left Channel')
    plt.plot(snr_levels, get_metric('snr_improvement_r'), 'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('SegSNR Improvement (dB)')
    plt.title('SegSNR Improvement by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "segsnr_improvement.png"))
    plt.close()
    
    # Interaural Cue Preservation
    plt.figure(figsize=(10, 6))
    plt.plot(snr_levels, get_metric('ild_error'), 'o-', label='ILD Error (dB)')
    plt.plot(snr_levels, [e/10 for e in get_metric('ipd_error')], 'o-', label='IPD Error (° ÷ 10)')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('Error')
    plt.title('Interaural Cue Preservation by SNR Level')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "interaural_errors.png"))
    plt.close()
    
    # Combined metrics overview
    plt.figure(figsize=(15, 10))
    
    # SegSNR subplot
    plt.subplot(2, 2, 1)
    plt.plot(snr_levels, get_metric('snr_improvement_l'), 'o-', label='Left Channel')
    plt.plot(snr_levels, get_metric('snr_improvement_r'), 'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('SegSNR Improvement (dB)')
    plt.title('Noise Reduction Performance')
    plt.grid(True)
    plt.legend()
    
    # STOI subplot
    plt.subplot(2, 2, 2)
    plt.plot(snr_levels, get_metric('stoi_improvement_l'), 'o-', label='Left Channel')
    plt.plot(snr_levels, get_metric('stoi_improvement_r'), 'o-', label='Right Channel')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('STOI Improvement')
    plt.title('Speech Intelligibility Improvement')
    plt.grid(True)
    plt.legend()
    
    # MBSTOI subplot
    plt.subplot(2, 2, 3)
    if MBSTOI_AVAILABLE:
        plt.plot(snr_levels, get_metric('mbstoi_improvement'), 'o-')
        plt.xlabel('SNR Level (dB)')
        plt.ylabel('MBSTOI Improvement')
        plt.title('Binaural Speech Intelligibility')
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'MBSTOI not available', 
                 horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
    
    # Interaural Cues subplot
    plt.subplot(2, 2, 4)
    plt.plot(snr_levels, get_metric('ild_error'), 'o-', label='ILD Error (dB)')
    plt.plot(snr_levels, [e/10 for e in get_metric('ipd_error')], 'o-', label='IPD Error (° ÷ 10)')
    plt.xlabel('SNR Level (dB)')
    plt.ylabel('Error')
    plt.title('Interaural Cue Preservation')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_overview.png"))
    plt.close()
    
def create_summary_report(all_results, output_dir):
    """Create a markdown report summarizing all results"""
    if not all_results:
        print("No results to include in report")
        return
    
    # Sort results by SNR level
    all_results.sort(key=lambda r: r['snr_level'])
    
    with open(os.path.join(output_dir, "summary_report.md"), 'w') as f:
        f.write("# Binaural Speech Enhancement Evaluation Results\n\n")
        
        f.write("## Overview\n")
        f.write("This report summarizes the performance of the BCCTN model across different SNR levels.\n\n")
        
        # STOI Table
        f.write("## Speech Intelligibility (STOI)\n\n")
        f.write("| SNR (dB) | Noisy (L) | Enhanced (L) | Improvement (L) | Noisy (R) | Enhanced (R) | Improvement (R) |\n")
        f.write("|----------|-----------|--------------|-----------------|-----------|--------------|------------------|\n")
        
        for result in all_results:
            metrics = result['metrics']
            snr = result['snr_level']
            
            f.write(f"| {snr} | {metrics['stoi_noisy_l']:.3f} | {metrics['stoi_enhanced_l']:.3f} | ")
            f.write(f"**{metrics['stoi_improvement_l']:.3f}** | {metrics['stoi_noisy_r']:.3f} | ")
            f.write(f"{metrics['stoi_enhanced_r']:.3f} | **{metrics['stoi_improvement_r']:.3f}** |\n")
        
        # MBSTOI Table
        if MBSTOI_AVAILABLE:
            f.write("\n## Binaural Speech Intelligibility (MBSTOI)\n\n")
            f.write("| SNR (dB) | Noisy | Enhanced | Improvement |\n")
            f.write("|----------|-------|----------|-------------|\n")
            
            for result in all_results:
                metrics = result['metrics']
                snr = result['snr_level']
                
                f.write(f"| {snr} | {metrics['mbstoi_noisy']:.3f} | {metrics['mbstoi_enhanced']:.3f} | ")
                f.write(f"**{metrics['mbstoi_improvement']:.3f}** |\n")
        
        # SegSNR Table
        f.write("\n## Noise Reduction (SegSNR in dB)\n\n")
        f.write("| SNR (dB) | Noisy (L) | Enhanced (L) | Improvement (L) | Noisy (R) | Enhanced (R) | Improvement (R) |\n")
        f.write("|----------|-----------|--------------|-----------------|-----------|--------------|------------------|\n")
        
        for result in all_results:
            metrics = result['metrics']
            snr = result['snr_level']
            
            f.write(f"| {snr} | {metrics['snr_noisy_l']:.2f} | {metrics['snr_enhanced_l']:.2f} | ")
            f.write(f"**{metrics['snr_improvement_l']:.2f}** | {metrics['snr_noisy_r']:.2f} | ")
            f.write(f"{metrics['snr_enhanced_r']:.2f} | **{metrics['snr_improvement_r']:.2f}** |\n")
        
        # Interaural Cues Table
        f.write("\n## Interaural Cue Preservation\n\n")
        f.write("| SNR (dB) | ILD Error (dB) | IPD Error (°) |\n")
        f.write("|----------|----------------|---------------|\n")
        
        for result in all_results:
            metrics = result['metrics']
            snr = result['snr_level']
            
            f.write(f"| {snr} | {metrics['ild_error']:.2f} | {metrics['ipd_error']:.2f} |\n")
        
        # Add visualizations to the report
        f.write("\n## Visualizations\n\n")
        f.write("### Overall Performance Metrics\n")
        f.write("![Metrics Overview](metrics_overview.png)\n\n")
        
        f.write("### SegSNR Improvement\n")
        f.write("![SegSNR Improvement](segsnr_improvement.png)\n\n")
        
        f.write("### STOI Improvement\n")
        f.write("![STOI Improvement](stoi_improvement.png)\n\n")
        
        if MBSTOI_AVAILABLE:
            f.write("### MBSTOI Improvement\n")
            f.write("![MBSTOI Improvement](mbstoi_improvement.png)\n\n")
        
        f.write("### Interaural Cue Preservation\n")
        f.write("![Interaural Errors](interaural_errors.png)\n")

def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate BCCTN model on different SNR levels")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory containing clean and noisy data")
    parser.add_argument("--clean_dir", type=str, help="Directory containing clean data (defaults to data_root/clean)")
    parser.add_argument("--noisy_dir", type=str, help="Directory containing noisy data (defaults to data_root/noisy)")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to Hydra config directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate per SNR level")
    parser.add_argument("--snr_levels", type=int, nargs="+", default=[-6, -3, 0, 3, 6, 9, 12, 15], 
                      help="SNR levels to evaluate (default: -6 -3 0 3 6 9 12 15)")
    parser.add_argument("--save_audio", action="store_true", help="Save audio samples for each SNR level")
    
    args = parser.parse_args()
    
    # Initialize Hydra config
    GlobalHydra.instance().clear()
    initialize(config_path=args.config_path)
    config = compose("config")
    
    # Set up directories
    if not args.clean_dir:
        args.clean_dir = os.path.join(args.data_root, "clean")
    if not args.noisy_dir:
        args.noisy_dir = os.path.join(args.data_root, "noisy")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if directories exist
    for path in [args.clean_dir, args.noisy_dir, args.checkpoint]:
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist.")
            return
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get state dict and check for needed modifications
    state_dict = checkpoint["state_dict"]
    
    # Check if model prefix needs to be added
    if all(not k.startswith("model.") for k in state_dict.keys()):
        # Add model. prefix
        state_dict = {"model." + k: v for k, v in state_dict.items()}
    
    # Now load with strict=False to allow for some flexibility
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    print(f"Model loaded successfully")
    
    # Evaluate each SNR level
    all_results = []
    for snr_level in args.snr_levels:
        result = evaluate_snr_level(
            model, args.clean_dir, args.noisy_dir, snr_level, 
            device, args.num_samples, args.save_audio, args.output_dir
        )
        if result:
            all_results.append(result)
    
    # Create summary visualizations
    if all_results:
        plot_metrics_by_snr(all_results, args.output_dir)
        create_summary_report(all_results, args.output_dir)
        
        # Also save all results as a CSV
        summary_data = {
            'SNR_Level': [],
            'STOI_Noisy_L': [], 'STOI_Enhanced_L': [], 'STOI_Improvement_L': [],
            'STOI_Noisy_R': [], 'STOI_Enhanced_R': [], 'STOI_Improvement_R': [],
            'MBSTOI_Noisy': [], 'MBSTOI_Enhanced': [], 'MBSTOI_Improvement': [],
            'SegSNR_Noisy_L': [], 'SegSNR_Enhanced_L': [], 'SegSNR_Improvement_L': [],
            'SegSNR_Noisy_R': [], 'SegSNR_Enhanced_R': [], 'SegSNR_Improvement_R': [],
            'ILD_Error': [], 'IPD_Error': []
        }
        
        for result in all_results:
            metrics = result['metrics']
            summary_data['SNR_Level'].append(result['snr_level'])
            
            # Add metrics to summary data
            summary_data['STOI_Noisy_L'].append(metrics['stoi_noisy_l'])
            summary_data['STOI_Enhanced_L'].append(metrics['stoi_enhanced_l'])
            summary_data['STOI_Improvement_L'].append(metrics['stoi_improvement_l'])
            summary_data['STOI_Noisy_R'].append(metrics['stoi_noisy_r'])
            summary_data['STOI_Enhanced_R'].append(metrics['stoi_enhanced_r'])
            summary_data['STOI_Improvement_R'].append(metrics['stoi_improvement_r'])
            
            summary_data['MBSTOI_Noisy'].append(metrics['mbstoi_noisy'])
            summary_data['MBSTOI_Enhanced'].append(metrics['mbstoi_enhanced'])
            summary_data['MBSTOI_Improvement'].append(metrics['mbstoi_improvement'])
            
            summary_data['SegSNR_Noisy_L'].append(metrics['snr_noisy_l'])
            summary_data['SegSNR_Enhanced_L'].append(metrics['snr_enhanced_l'])
            summary_data['SegSNR_Improvement_L'].append(metrics['snr_improvement_l'])
            summary_data['SegSNR_Noisy_R'].append(metrics['snr_noisy_r'])
            summary_data['SegSNR_Enhanced_R'].append(metrics['snr_enhanced_r'])
            summary_data['SegSNR_Improvement_R'].append(metrics['snr_improvement_r'])
            
            summary_data['ILD_Error'].append(metrics['ild_error'])
            summary_data['IPD_Error'].append(metrics['ipd_error'])
        
        # Save as CSV
        pd.DataFrame(summary_data).to_csv(os.path.join(args.output_dir, "all_results.csv"), index=False)
        
        print(f"\nEvaluation completed successfully. Results saved to {args.output_dir}")
        print(f"Summary report: {os.path.join(args.output_dir, 'summary_report.md')}")
        print(f"Summary CSV: {os.path.join(args.output_dir, 'all_results.csv')}")
    else:
        print("No valid results were obtained. Please check your data directories and SNR levels.")

if __name__ == "__main__":
    main()