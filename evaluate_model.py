#!/usr/bin/env python
# coding: utf-8

import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
from tqdm import tqdm
from pathlib import Path
import sys

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

from DCNN.trainer import DCNNLightningModule
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.feature_extractors import Stft, IStft
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

# Try to import PESQ - it might not be installed
try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    print("Warning: PESQ not available. Install it with 'pip install pesq'")
    PESQ_AVAILABLE = False

try:
    # Try direct import from MBSTOI package
    from MBSTOI.mbstoi import mbstoi
    print("Using MBSTOI from MBSTOI package")
    MBSTOI_AVAILABLE = True
except ImportError:
    try:
        # Try with relative import 
        import sys
        import os
        # Add parent directory to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from MBSTOI.mbstoi import mbstoi
        print("Using MBSTOI with relative import")
        MBSTOI_AVAILABLE = True
    except ImportError:
        print("Warning: MBSTOI not available. Install it with proper dependencies.")
        MBSTOI_AVAILABLE = False


class Config:
    fs = 16000  # Default sampling rate

CONFIG = Config()


def prepare_for_stft(x):
    """Convert 1D array to 3D tensor with shape (batch=1, channel=1, time)"""
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0)

def calculate_mbstoi(clean_l, clean_r, proc_l, proc_r):
    """Unified MBSTOI calculation with error handling"""
    if not MBSTOI_AVAILABLE:
        print("MBSTOI calculation not available, returning 0")
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
        
        # Call the actual MBSTOI function
        result = mbstoi(clean_l, clean_r, proc_l, proc_r)
        return result
    except Exception as e:
        print(f"Error in MBSTOI calculation: {e}")
        return 0


def compute_ild_db(s1, s2, eps=1e-6):
    """Compute Interaural Level Difference in dB"""
    l1 = 20 * np.log10(np.abs(s1) + eps)
    l2 = 20 * np.log10(np.abs(s2) + eps)
    return l1 - l2

def compute_ipd_rad(s1, s2, eps=1e-6):
    """Compute Interaural Phase Difference in radians correctly"""
    # Calculate phase difference and wrap to [-π, π]
    phase_diff = np.angle(s1 * np.conj(s2) + eps)
    return phase_diff

def speech_mask(stft_l, stft_r, threshold=40):  # Increased threshold to 40dB
    """Create a speech binary mask with proper threshold"""
    # Calculate energy (in dB) from both channels
    energy = 10 * np.log10(np.abs(stft_l)**2 + np.abs(stft_r)**2 + 1e-12)
    
    # Compute maximum energy per frame (time)
    max_per_frame = np.max(energy, axis=0, keepdims=True)
    
    # Create mask: select bins above threshold relative to max
    mask = (energy >= max_per_frame - threshold)
    
    # Print mask statistics for debugging
    active_bins = np.sum(mask)
    total_bins = mask.size
    print(f"Speech mask active bins: {active_bins}/{total_bins} ({active_bins/total_bins:.2%})")
    
    return mask


def calculate_ild_loss(target_stft_l, target_stft_r, output_stft_l, output_stft_r):
    """Calculate ILD loss between target and output signals"""
    # Calculate ILDs
    target_ild = compute_ild_db(target_stft_l, target_stft_r)
    output_ild = compute_ild_db(output_stft_l, output_stft_r)
    
    # Create a speech activity mask
    mask = speech_mask(target_stft_l, target_stft_r, threshold=20)
    
    # Print mask statistics for debugging
    active_bins = np.sum(mask)
    total_bins = mask.size
    print(f"Speech mask active bins: {active_bins}/{total_bins} ({active_bins/total_bins:.2%})")
    
    # Calculate the absolute difference and apply mask
    ild_error = np.abs(target_ild - output_ild)
    masked_ild_error = ild_error * mask
    
    # Average over speech-active regions
    if np.sum(mask) > 0:
        return np.sum(masked_ild_error) / np.sum(mask)
    else:
        print("Warning: Empty speech mask in ILD calculation")
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


# def calculate_snr(clean, processed, max_delay=64):  # Increased from 10 to 64
#     """Calculate Signal-to-Noise Ratio in dB with proper alignment"""
#     # Ensure same length
#     min_len = min(len(clean), len(processed))
#     clean_temp = clean[:min_len].copy()
#     processed_temp = processed[:min_len].copy()
    
#     # Find optimal alignment using cross-correlation
#     if max_delay > 0:
#         # Compute cross-correlation
#         corr = np.correlate(clean_temp, processed_temp, mode='full')
#         max_idx = np.argmax(np.abs(corr))
#         center = len(corr) // 2
#         delay = max_idx - center
        
#         # Limit to max_delay
#         delay = max(min(delay, max_delay), -max_delay)
        
#         # Apply delay
#         if delay > 0:
#             # processed is delayed
#             clean = clean_temp[delay:]
#             processed = processed_temp[:-delay] if delay > 0 else processed_temp
#         else:
#             # clean is delayed
#             clean = clean_temp[:delay] if delay < 0 else clean_temp
#             processed = processed_temp[-delay:]
            
#         # Ensure same length after alignment
#         min_len = min(len(clean), len(processed))
#         clean = clean[:min_len]
#         processed = processed[:min_len]
#     else:
#         clean = clean_temp
#         processed = processed_temp
    
#     # Calculate energy of clean signal
#     clean_energy = np.sum(clean**2) + 1e-10
    
#     # Calculate MSE between clean and processed
#     noise = processed - clean
#     noise_energy = np.sum(noise**2) + 1e-10
    
#     # Calculate SNR
#     snr = 10 * np.log10(clean_energy / noise_energy)
    
#     # Cap extreme SNRs
#     return max(min(snr, 30), -30)

def calculate_seg_snr(clean, processed, frame_size=256, hop=128, max_delay=64):
    """Calculate frequency-weighted Segmental SNR"""
    # Align signals first
    min_len = min(len(clean), len(processed))
    clean_temp = clean[:min_len].copy()
    processed_temp = processed[:min_len].copy()
    
    # Find optimal alignment using cross-correlation
    corr = np.correlate(clean_temp, processed_temp, mode='full')
    max_idx = np.argmax(np.abs(corr))
    center = len(corr) // 2
    delay = max_idx - center
    
    # Limit to max_delay
    delay = max(min(delay, max_delay), -max_delay)
    
    # Apply delay
    if delay > 0:
        clean_aligned = clean_temp[delay:]
        processed_aligned = processed_temp[:-delay] if delay > 0 else processed_temp
    else:
        clean_aligned = clean_temp[:delay] if delay < 0 else clean_temp
        processed_aligned = processed_temp[-delay:]
        
    # Ensure same length after alignment
    min_len = min(len(clean_aligned), len(processed_aligned))
    clean_aligned = clean_aligned[:min_len]
    processed_aligned = processed_aligned[:min_len]
    
    # Split into frames
    num_frames = (min_len - frame_size) // hop + 1
    seg_snrs = []
    
    for i in range(num_frames):
        start = i * hop
        clean_frame = clean_aligned[start:start+frame_size]
        proc_frame = processed_aligned[start:start+frame_size]
        
        # Calculate frame energy and noise energy
        clean_energy = np.sum(clean_frame**2) + 1e-10
        noise = proc_frame - clean_frame
        noise_energy = np.sum(noise**2) + 1e-10
        
        # Calculate SNR for this frame
        frame_snr = 10 * np.log10(clean_energy / noise_energy)
        
        # Cap extreme values
        frame_snr = max(min(frame_snr, 35), -10)
        seg_snrs.append(frame_snr)
    
    # Return average and aligned signals
    return np.mean(seg_snrs), clean_aligned, processed_aligned

def calculate_snr_and_align(clean, processed, max_delay=64):
    """Calculate SNR and return aligned signals"""
    # Ensure same length
    min_len = min(len(clean), len(processed))
    clean_temp = clean[:min_len].copy()
    processed_temp = processed[:min_len].copy()
    
    # Find optimal alignment using cross-correlation
    if max_delay > 0:
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
    else:
        clean_aligned = clean_temp
        processed_aligned = processed_temp
    
    # Calculate energy of clean signal
    clean_energy = np.sum(clean_aligned**2) + 1e-10
    
    # Calculate MSE between clean and processed
    noise = processed_aligned - clean_aligned
    noise_energy = np.sum(noise**2) + 1e-10
    
    # Calculate SNR
    snr = 10 * np.log10(clean_energy / noise_energy)
    
    # Cap extreme SNRs
    snr = max(min(snr, 30), -30)
    
    return snr, clean_aligned, processed_aligned

def evaluate_model(config_path="./config", 
                  model_checkpoint_path=None,
                  noisy_dataset_path=None, 
                  clean_dataset_path=None,
                  output_dir="evaluation_results", 
                  num_samples=100):
    """
    Comprehensive evaluation of the binaural speech enhancement model with multiple metrics
    
    Args:
        config_path: Path to the config directory
        model_checkpoint_path: Path to the saved model checkpoint
        noisy_dataset_path: Path to noisy test dataset, if None uses config value
        clean_dataset_path: Path to clean test dataset, if None uses config value 
        output_dir: Directory to save visualizations and results
        num_samples: Number of samples to evaluate
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Hydra config
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    config = compose("config")
    
    # If paths not provided, use from config
    if noisy_dataset_path is None:
        noisy_dataset_path = config.dataset.noisy_test_dataset_dir
    if clean_dataset_path is None:
        clean_dataset_path = config.dataset.target_test_dataset_dir
    
    # If model_checkpoint_path not provided, use default
    if model_checkpoint_path is None:
        model_checkpoint_path = "DCNN/Checkpoints/Trained_model.ckpt"
    
    # Check if paths exist
    for path in [noisy_dataset_path, clean_dataset_path, model_checkpoint_path]:
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist.")
            return
    
    print(f"Loading model from {model_checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = DCNNLightningModule(config)
    model.eval()
    torch.cuda.empty_cache()  # Clear CUDA cache between samples

    # Load checkpoint with modified state dict handling
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    
    # Get state dict and check for needed modifications
    state_dict = checkpoint["state_dict"]
    
    # Check if model prefix needs to be added
    # The loaded state dict has keys without 'model.' prefix, but the model expects them with it
    if all(not k.startswith("model.") for k in state_dict.keys()):
        # Instead of stripping "model." prefix, we need to add it
        state_dict = {"model." + k: v for k, v in state_dict.items()}
    
    # Now load with strict=False to allow for some flexibility
    model.load_state_dict(state_dict, strict=False)
    
    # Alternatively, if you're sure the structure is right but just prefixes are wrong:
    # model.model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model = model.to(device)

    # # Alternative approach
    # model = DCNNLightningModule(config)
    # checkpoint = torch.load(model_checkpoint_path, map_location=device)
    # model.model.load_state_dict(checkpoint["state_dict"], strict=False)
    # model = model.to(device)
    
    print(f"Model loaded successfully. Starting evaluation on {num_samples} samples...")
    
    # Initialize audio processing tools
    SR = 16000  # Sample rate
    win_len = 400
    win_inc = 100
    fft_len = 512
    
    stft = Stft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
    istft = IStft(n_dft=fft_len, hop_size=win_inc, win_length=win_len)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=SR)
    
    # Create dataset and dataloader
    dataset = BaseDataset(noisy_dataset_path, clean_dataset_path, mono=False)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=False,
    #     num_workers=2
    # )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,  # Keep batch size at 1 for evaluation
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=1  # Reduce worker count to avoid memory issues
    )
    
    # Initialize metrics storage
    metrics = {
        'stoi_noisy_l': [],
        'stoi_noisy_r': [],
        'stoi_enhanced_l': [],
        'stoi_enhanced_r': [],
        'mbstoi_noisy': [],
        'mbstoi_enhanced': [],
        'pesq_noisy_l': [],
        'pesq_noisy_r': [],
        'pesq_enhanced_l': [],
        'pesq_enhanced_r': [],
        'snr_noisy_l': [],
        'snr_noisy_r': [],
        'snr_enhanced_l': [],
        'snr_enhanced_r': [],
        'ild_error': [],
        'ipd_error': []
    }
    
    # with torch.no_grad():
    #     for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
    #         if i >= num_samples:
    #             break
                
    #         # Get data
    #         noisy_samples = batch[0].to(device)
    #         clean_samples = batch[1].to(device)
            
    #         # Process with model
    #         model_output = model(noisy_samples).cpu()
            
    #         # Move tensors to CPU for analysis
    #         noisy_samples = noisy_samples.cpu()
    #         clean_samples = clean_samples.cpu()
            
    #         # Convert to numpy for easier processing
    #         noisy_np = noisy_samples[0].numpy()
    #         clean_np = clean_samples[0].numpy()
    #         enhanced_np = model_output[0].numpy()
            
    #         # # Calculate STOI
    #         # metrics['stoi_noisy_l'].append(stoi_metric(torch.from_numpy(noisy_np[0]), torch.from_numpy(clean_np[0])).item())
    #         # metrics['stoi_noisy_r'].append(stoi_metric(torch.from_numpy(noisy_np[1]), torch.from_numpy(clean_np[1])).item())
    #         # metrics['stoi_enhanced_l'].append(stoi_metric(torch.from_numpy(enhanced_np[0]), torch.from_numpy(clean_np[0])).item())
    #         # metrics['stoi_enhanced_r'].append(stoi_metric(torch.from_numpy(enhanced_np[1]), torch.from_numpy(clean_np[1])).item())

    #         # Calculate SNR for noisy signals and keep aligned versions
    #         snr_noisy_l, clean_aligned_noisy_l, noisy_aligned_l = calculate_snr_and_align(clean_np[0], noisy_np[0])
    #         snr_noisy_r, clean_aligned_noisy_r, noisy_aligned_r = calculate_snr_and_align(clean_np[1], noisy_np[1])

    #         # Calculate SNR for enhanced signals and keep aligned versions
    #         snr_enhanced_l, clean_aligned_enh_l, enhanced_aligned_l = calculate_snr_and_align(clean_np[0], enhanced_np[0])
    #         snr_enhanced_r, clean_aligned_enh_r, enhanced_aligned_r = calculate_snr_and_align(clean_np[1], enhanced_np[1])

    #         # Store SNR values
    #         metrics['snr_noisy_l'].append(snr_noisy_l)
    #         metrics['snr_noisy_r'].append(snr_noisy_r)
    #         metrics['snr_enhanced_l'].append(snr_enhanced_l)
    #         metrics['snr_enhanced_r'].append(snr_enhanced_r)

    #         # Use aligned signals for STOI calculation
    #         metrics['stoi_noisy_l'].append(stoi_metric(torch.from_numpy(clean_aligned_noisy_l), 
    #                                                 torch.from_numpy(noisy_aligned_l)).item())
    #         metrics['stoi_noisy_r'].append(stoi_metric(torch.from_numpy(clean_aligned_noisy_r), 
    #                                                 torch.from_numpy(noisy_aligned_r)).item())
    #         metrics['stoi_enhanced_l'].append(stoi_metric(torch.from_numpy(clean_aligned_enh_l), 
    #                                                 torch.from_numpy(enhanced_aligned_l)).item())
    #         metrics['stoi_enhanced_r'].append(stoi_metric(torch.from_numpy(clean_aligned_enh_r), 
    #                                                 torch.from_numpy(enhanced_aligned_r)).item())
            
    #         # # Calculate MBSTOI if available
    #         # if MBSTOI_AVAILABLE:
    #         #     try:
    #         #         metrics['mbstoi_noisy'].append(mbstoi(clean_np[0], clean_np[1], noisy_np[0], noisy_np[1]))
    #         #         metrics['mbstoi_enhanced'].append(mbstoi(clean_np[0], clean_np[1], enhanced_np[0], enhanced_np[1]))
    #         #     except Exception as e:
    #         #         print(f"Error calculating MBSTOI: {e}")
    #         #         metrics['mbstoi_noisy'].append(0)
    #         #         metrics['mbstoi_enhanced'].append(0)

    #         # Calculate MBSTOI if available
    #         if MBSTOI_AVAILABLE:
    #             try:
    #                 # Use aligned signals for MBSTOI calculation
    #                 metrics['mbstoi_noisy'].append(mbstoi(clean_aligned_noisy_l, clean_aligned_noisy_r, 
    #                                                     noisy_aligned_l, noisy_aligned_r))
    #                 metrics['mbstoi_enhanced'].append(mbstoi(clean_aligned_enh_l, clean_aligned_enh_r, 
    #                                                     enhanced_aligned_l, enhanced_aligned_r))
    #             except Exception as e:
    #                 print(f"Error calculating MBSTOI: {e}")
    #                 metrics['mbstoi_noisy'].append(0)
    #                 metrics['mbstoi_enhanced'].append(0)
            
    #         # Calculate PESQ if available
    #         # if PESQ_AVAILABLE:
    #         #     try:
    #         #         metrics['pesq_noisy_l'].append(pesq(SR, clean_np[0], noisy_np[0], 'wb'))
    #         #         metrics['pesq_noisy_r'].append(pesq(SR, clean_np[1], noisy_np[1], 'wb'))
    #         #         metrics['pesq_enhanced_l'].append(pesq(SR, clean_np[0], enhanced_np[0], 'wb'))
    #         #         metrics['pesq_enhanced_r'].append(pesq(SR, clean_np[1], enhanced_np[1], 'wb'))
    #         #     except Exception as e:
    #         #         print(f"Error calculating PESQ: {e}")
    #         #         metrics['pesq_noisy_l'].append(0)
    #         #         metrics['pesq_noisy_r'].append(0)
    #         #         metrics['pesq_enhanced_l'].append(0)
    #         #         metrics['pesq_enhanced_r'].append(0)

    #         # Calculate PESQ if available using aligned signals
    #         if PESQ_AVAILABLE:
    #             try:
    #                 metrics['pesq_noisy_l'].append(pesq(SR, clean_aligned_noisy_l, noisy_aligned_l, 'wb'))
    #                 metrics['pesq_noisy_r'].append(pesq(SR, clean_aligned_noisy_r, noisy_aligned_r, 'wb'))
    #                 metrics['pesq_enhanced_l'].append(pesq(SR, clean_aligned_enh_l, enhanced_aligned_l, 'wb'))
    #                 metrics['pesq_enhanced_r'].append(pesq(SR, clean_aligned_enh_r, enhanced_aligned_r, 'wb'))
    #             except Exception as e:
    #                 print(f"Error calculating PESQ: {e}")
    #                 metrics['pesq_noisy_l'].append(0)
    #                 metrics['pesq_noisy_r'].append(0)
    #                 metrics['pesq_enhanced_l'].append(0)
    #                 metrics['pesq_enhanced_r'].append(0)
            
    #         # # Calculate SNR
    #         # metrics['snr_noisy_l'].append(calculate_snr(clean_np[0], noisy_np[0]))
    #         # metrics['snr_noisy_r'].append(calculate_snr(clean_np[1], noisy_np[1]))
    #         # metrics['snr_enhanced_l'].append(calculate_snr(clean_np[0], enhanced_np[0]))
    #         # metrics['snr_enhanced_r'].append(calculate_snr(clean_np[1], enhanced_np[1]))
        

    #         noisy_stft_l = stft(prepare_for_stft(noisy_np[0])).squeeze(0).numpy()
    #         noisy_stft_r = stft(prepare_for_stft(noisy_np[1])).squeeze(0).numpy()
    #         enhanced_stft_l = stft(prepare_for_stft(enhanced_np[0])).squeeze(0).numpy()
    #         enhanced_stft_r = stft(prepare_for_stft(enhanced_np[1])).squeeze(0).numpy()
    #         clean_stft_l = stft(prepare_for_stft(clean_np[0])).squeeze(0).numpy()
    #         clean_stft_r = stft(prepare_for_stft(clean_np[1])).squeeze(0).numpy()
            
    #         # Calculate ILD and IPD errors
    #         ild_error = calculate_ild_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
    #         ipd_error = calculate_ipd_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
            
    #         metrics['ild_error'].append(ild_error)
    #         metrics['ipd_error'].append(ipd_error)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
            if i >= num_samples:
                break
                
            # Get data
            noisy_samples = batch[0].to(device)
            clean_samples = batch[1].to(device)
            
            # Process with model
            model_output = model(noisy_samples).cpu()
            
            # Move tensors to CPU for analysis
            noisy_samples = noisy_samples.cpu()
            clean_samples = clean_samples.cpu()
            
            # Convert to numpy for easier processing
            noisy_np = noisy_samples[0].numpy()
            clean_np = clean_samples[0].numpy()
            enhanced_np = model_output[0].numpy()
            
            # Calculate SegSNR and get aligned signals
            seg_snr_noisy_l, clean_aligned_noisy_l, noisy_aligned_l = calculate_seg_snr(clean_np[0], noisy_np[0])
            seg_snr_noisy_r, clean_aligned_noisy_r, noisy_aligned_r = calculate_seg_snr(clean_np[1], noisy_np[1])
            seg_snr_enh_l, clean_aligned_enh_l, enhanced_aligned_l = calculate_seg_snr(clean_np[0], enhanced_np[0])
            seg_snr_enh_r, clean_aligned_enh_r, enhanced_aligned_r = calculate_seg_snr(clean_np[1], enhanced_np[1])
            
            # Store SegSNR values
            metrics['snr_noisy_l'].append(seg_snr_noisy_l)
            metrics['snr_noisy_r'].append(seg_snr_noisy_r)
            metrics['snr_enhanced_l'].append(seg_snr_enh_l)
            metrics['snr_enhanced_r'].append(seg_snr_enh_r)
            
            # Calculate STOI using aligned signals
            metrics['stoi_noisy_l'].append(stoi_metric(torch.from_numpy(clean_aligned_noisy_l), 
                                                    torch.from_numpy(noisy_aligned_l)).item())
            metrics['stoi_noisy_r'].append(stoi_metric(torch.from_numpy(clean_aligned_noisy_r), 
                                                    torch.from_numpy(noisy_aligned_r)).item())
            metrics['stoi_enhanced_l'].append(stoi_metric(torch.from_numpy(clean_aligned_enh_l), 
                                                        torch.from_numpy(enhanced_aligned_l)).item())
            metrics['stoi_enhanced_r'].append(stoi_metric(torch.from_numpy(clean_aligned_enh_r), 
                                                        torch.from_numpy(enhanced_aligned_r)).item())
            
            # Calculate MBSTOI using aligned signals
            if MBSTOI_AVAILABLE:
                try:
                    metrics['mbstoi_noisy'].append(mbstoi(clean_aligned_noisy_l, clean_aligned_noisy_r, 
                                                        noisy_aligned_l, noisy_aligned_r))
                    metrics['mbstoi_enhanced'].append(mbstoi(clean_aligned_enh_l, clean_aligned_enh_r, 
                                                            enhanced_aligned_l, enhanced_aligned_r))
                except Exception as e:
                    print(f"Error calculating MBSTOI: {e}")
                    metrics['mbstoi_noisy'].append(0)
                    metrics['mbstoi_enhanced'].append(0)
          
            if PESQ_AVAILABLE:
                try:
                    # Make sure we're using the aligned signals
                    clean_l, noisy_l = clean_aligned_noisy_l, noisy_aligned_l
                    clean_r, noisy_r = clean_aligned_noisy_r, noisy_aligned_r
                    enhanced_l, enhanced_r = enhanced_aligned_l, enhanced_aligned_r
                    
                    # Calculate PESQ
                    metrics['pesq_noisy_l'].append(pesq(SR, clean_l, noisy_l, 'wb'))
                    metrics['pesq_noisy_r'].append(pesq(SR, clean_r, noisy_r, 'wb'))
                    metrics['pesq_enhanced_l'].append(pesq(SR, clean_l, enhanced_l, 'wb'))
                    metrics['pesq_enhanced_r'].append(pesq(SR, clean_r, enhanced_r, 'wb'))
                    
                    print(f"PESQ Calculated - Noisy: {metrics['pesq_noisy_l'][-1]:.2f}/{metrics['pesq_noisy_r'][-1]:.2f}, "
                        f"Enhanced: {metrics['pesq_enhanced_l'][-1]:.2f}/{metrics['pesq_enhanced_r'][-1]:.2f}")
                except Exception as e:
                    print(f"Error calculating PESQ: {e}")
                    # Initialize these lists if they don't exist
                    for key in ['pesq_noisy_l', 'pesq_noisy_r', 'pesq_enhanced_l', 'pesq_enhanced_r']:
                        if key not in metrics:
                            metrics[key] = []
                        metrics[key].append(0)            
            
            # Use the proper STFT for all signals
            noisy_stft_l = stft(prepare_for_stft(noisy_aligned_l)).squeeze(0).numpy()
            noisy_stft_r = stft(prepare_for_stft(noisy_aligned_r)).squeeze(0).numpy()
            enhanced_stft_l = stft(prepare_for_stft(enhanced_aligned_l)).squeeze(0).numpy()
            enhanced_stft_r = stft(prepare_for_stft(enhanced_aligned_r)).squeeze(0).numpy()
            clean_stft_l = stft(prepare_for_stft(clean_aligned_enh_l)).squeeze(0).numpy()
            clean_stft_r = stft(prepare_for_stft(clean_aligned_enh_r)).squeeze(0).numpy()
            
            # Calculate ILD and IPD errors
            ild_error = calculate_ild_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
            ipd_error = calculate_ipd_loss(clean_stft_l, clean_stft_r, enhanced_stft_l, enhanced_stft_r)
            
            metrics['ild_error'].append(ild_error)
            metrics['ipd_error'].append(ipd_error)

            # Save audio files
            save_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(save_dir, exist_ok=True)
            
            sf.write(os.path.join(save_dir, "noisy_L.wav"), noisy_np[0], SR)
            sf.write(os.path.join(save_dir, "noisy_R.wav"), noisy_np[1], SR)
            sf.write(os.path.join(save_dir, "enhanced_L.wav"), enhanced_np[0], SR)
            sf.write(os.path.join(save_dir, "enhanced_R.wav"), enhanced_np[1], SR)
            sf.write(os.path.join(save_dir, "clean_L.wav"), clean_np[0], SR)
            sf.write(os.path.join(save_dir, "clean_R.wav"), clean_np[1], SR)
            
            # Create visualizations
            # 1. Spectrograms
            plt.figure(figsize=(18, 12))
            
            # Left channel spectrograms
            plt.subplot(3, 2, 1)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[0])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Noisy Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 2, 3)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[0])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Enhanced Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 2, 5)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[0])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Clean Left Channel')
            plt.colorbar(format='%+2.0f dB')
            
            # Right channel spectrograms
            plt.subplot(3, 2, 2)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[1])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Noisy Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 2, 4)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[1])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Enhanced Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(3, 2, 6)
            D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[1])), ref=np.max)
            librosa.display.specshow(D, y_axis='log', x_axis='time', sr=SR)
            plt.title('Clean Right Channel')
            plt.colorbar(format='%+2.0f dB')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "spectrograms.png"))
            plt.close()
            
            # 2. Interaural Cues Visualization
            plt.figure(figsize=(15, 10))
            
            # ILD visualization
            plt.subplot(2, 2, 1)
            plt.imshow(compute_ild_db(clean_stft_l, clean_stft_r), aspect='auto', origin='lower', cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Clean ILD (dB)')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
            
            plt.subplot(2, 2, 2)
            plt.imshow(compute_ild_db(enhanced_stft_l, enhanced_stft_r), aspect='auto', origin='lower', cmap='coolwarm')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Enhanced ILD (dB)')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
            
            # IPD visualization
            plt.subplot(2, 2, 3)
            plt.imshow(np.degrees(compute_ipd_rad(clean_stft_l, clean_stft_r)), aspect='auto', origin='lower', cmap='hsv', vmin=-180, vmax=180)
            plt.colorbar(format='%+2.0f°')
            plt.title('Clean IPD (degrees)')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
            
            plt.subplot(2, 2, 4)
            plt.imshow(np.degrees(compute_ipd_rad(enhanced_stft_l, enhanced_stft_r)), aspect='auto', origin='lower', cmap='hsv', vmin=-180, vmax=180)
            plt.colorbar(format='%+2.0f°')
            plt.title('Enhanced IPD (degrees)')
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "interaural_cues.png"))
            plt.close()
            
            # Print current sample metrics
            print(f"\nSample {i} Metrics:")
            print(f"STOI (L/R): Noisy={metrics['stoi_noisy_l'][-1]:.3f}/{metrics['stoi_noisy_r'][-1]:.3f}, "
                  f"Enhanced={metrics['stoi_enhanced_l'][-1]:.3f}/{metrics['stoi_enhanced_r'][-1]:.3f}")
            if MBSTOI_AVAILABLE:
                print(f"MBSTOI: Noisy={metrics['mbstoi_noisy'][-1]:.3f}, Enhanced={metrics['mbstoi_enhanced'][-1]:.3f}")
            if PESQ_AVAILABLE:
                if 'pesq_noisy_l' in metrics and metrics['pesq_noisy_l'] and 'pesq_noisy_r' in metrics and metrics['pesq_noisy_r']:
                    print(f"PESQ (L/R): Noisy={metrics['pesq_noisy_l'][-1]:.2f}/{metrics['pesq_noisy_r'][-1]:.2f}, "
                        f"Enhanced={metrics['pesq_enhanced_l'][-1]:.2f}/{metrics['pesq_enhanced_r'][-1]:.2f}")
                else:
                    print("PESQ metrics not available")
            print(f"SNR (L/R): Noisy={metrics['snr_noisy_l'][-1]:.2f}/{metrics['snr_noisy_r'][-1]:.2f} dB, "
                  f"Enhanced={metrics['snr_enhanced_l'][-1]:.2f}/{metrics['snr_enhanced_r'][-1]:.2f} dB")
            print(f"ILD Error: {metrics['ild_error'][-1]:.2f} dB")
            print(f"IPD Error: {metrics['ipd_error'][-1]:.2f} degrees")
    
    # Calculate average metrics
    avg_metrics = {}
    for key in metrics:
        if len(metrics[key]) > 0:
            avg_metrics[key] = np.mean(metrics[key])
    
    # Print average metrics
    print("\n=== Average Metrics ===")
    print(f"STOI (L/R): Noisy={avg_metrics.get('stoi_noisy_l', 0):.3f}/{avg_metrics.get('stoi_noisy_r', 0):.3f}, "
          f"Enhanced={avg_metrics.get('stoi_enhanced_l', 0):.3f}/{avg_metrics.get('stoi_enhanced_r', 0):.3f}")
    print(f"STOI Improvement (L/R): {avg_metrics.get('stoi_enhanced_l', 0) - avg_metrics.get('stoi_noisy_l', 0):.3f}/"
          f"{avg_metrics.get('stoi_enhanced_r', 0) - avg_metrics.get('stoi_noisy_r', 0):.3f}")
    
    if MBSTOI_AVAILABLE:
        print(f"MBSTOI: Noisy={avg_metrics.get('mbstoi_noisy', 0):.3f}, Enhanced={avg_metrics.get('mbstoi_enhanced', 0):.3f}")
        print(f"MBSTOI Improvement: {avg_metrics.get('mbstoi_enhanced', 0) - avg_metrics.get('mbstoi_noisy', 0):.3f}")
    
    if PESQ_AVAILABLE:
        print(f"PESQ (L/R): Noisy={avg_metrics.get('pesq_noisy_l', 0):.2f}/{avg_metrics.get('pesq_noisy_r', 0):.2f}, "
              f"Enhanced={avg_metrics.get('pesq_enhanced_l', 0):.2f}/{avg_metrics.get('pesq_enhanced_r', 0):.2f}")
        print(f"PESQ Improvement (L/R): {avg_metrics.get('pesq_enhanced_l', 0) - avg_metrics.get('pesq_noisy_l', 0):.2f}/"
              f"{avg_metrics.get('pesq_enhanced_r', 0) - avg_metrics.get('pesq_noisy_r', 0):.2f}")
    
    print(f"SNR (L/R): Noisy={avg_metrics.get('snr_noisy_l', 0):.2f}/{avg_metrics.get('snr_noisy_r', 0):.2f} dB, "
          f"Enhanced={avg_metrics.get('snr_enhanced_l', 0):.2f}/{avg_metrics.get('snr_enhanced_r', 0):.2f} dB")
    print(f"SNR Improvement (L/R): {avg_metrics.get('snr_enhanced_l', 0) - avg_metrics.get('snr_noisy_l', 0):.2f}/"
          f"{avg_metrics.get('snr_enhanced_r', 0) - avg_metrics.get('snr_noisy_r', 0):.2f} dB")
    
    print(f"ILD Error: {avg_metrics.get('ild_error', 0):.2f} dB")
    print(f"IPD Error: {avg_metrics.get('ipd_error', 0):.2f} degrees")
    
    # Create summary visualization
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.bar(['Noisy L', 'Noisy R', 'Enhanced L', 'Enhanced R'], 
            [avg_metrics.get('stoi_noisy_l', 0), avg_metrics.get('stoi_noisy_r', 0), 
             avg_metrics.get('stoi_enhanced_l', 0), avg_metrics.get('stoi_enhanced_r', 0)])
    plt.ylabel('STOI')
    plt.title('Speech Intelligibility')
    plt.ylim(0, 1)
    
    plt.subplot(2, 2, 2)
    if MBSTOI_AVAILABLE:
        plt.bar(['Noisy', 'Enhanced'], 
                [avg_metrics.get('mbstoi_noisy', 0), avg_metrics.get('mbstoi_enhanced', 0)])
        plt.ylabel('MBSTOI')
        plt.title('Binaural Speech Intelligibility')
        plt.ylim(0, 1)
    
    plt.subplot(2, 2, 3)
    plt.bar(['Noisy L', 'Noisy R', 'Enhanced L', 'Enhanced R'], 
            [avg_metrics.get('snr_noisy_l', 0), avg_metrics.get('snr_noisy_r', 0), 
             avg_metrics.get('snr_enhanced_l', 0), avg_metrics.get('snr_enhanced_r', 0)])
    plt.ylabel('SNR (dB)')
    plt.title('Signal-to-Noise Ratio')
    
    plt.subplot(2, 2, 4)
    plt.bar(['ILD Error', 'IPD Error'], 
            [avg_metrics.get('ild_error', 0), avg_metrics.get('ipd_error', 0)/10])  # Dividing IPD by 10 for scale
    plt.ylabel('Error (dB / 10° for IPD)')
    plt.title('Interaural Cue Preservation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_metrics.png"))
    plt.close()
    
    # Save metrics to file
    import pandas as pd
    
    # Save raw metrics
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "detailed_metrics.csv"), index=False)
    
    # Save summary metrics
    summary_metrics = {
        'Metric': [
            'STOI Noisy (L)', 'STOI Noisy (R)', 'STOI Enhanced (L)', 'STOI Enhanced (R)', 'STOI Improvement (L)', 'STOI Improvement (R)',
            'MBSTOI Noisy', 'MBSTOI Enhanced', 'MBSTOI Improvement',
            'PESQ Noisy (L)', 'PESQ Noisy (R)', 'PESQ Enhanced (L)', 'PESQ Enhanced (R)', 'PESQ Improvement (L)', 'PESQ Improvement (R)',
            'SNR Noisy (L)', 'SNR Noisy (R)', 'SNR Enhanced (L)', 'SNR Enhanced (R)', 'SNR Improvement (L)', 'SNR Improvement (R)',
            'ILD Error', 'IPD Error'
        ],
        'Value': [
            avg_metrics.get('stoi_noisy_l', 0), avg_metrics.get('stoi_noisy_r', 0), 
            avg_metrics.get('stoi_enhanced_l', 0), avg_metrics.get('stoi_enhanced_r', 0),
            avg_metrics.get('stoi_enhanced_l', 0) - avg_metrics.get('stoi_noisy_l', 0),
            avg_metrics.get('stoi_enhanced_r', 0) - avg_metrics.get('stoi_noisy_r', 0),
            
            avg_metrics.get('mbstoi_noisy', 0), avg_metrics.get('mbstoi_enhanced', 0),
            avg_metrics.get('mbstoi_enhanced', 0) - avg_metrics.get('mbstoi_noisy', 0),
            
            avg_metrics.get('pesq_noisy_l', 0), avg_metrics.get('pesq_noisy_r', 0),
            avg_metrics.get('pesq_enhanced_l', 0), avg_metrics.get('pesq_enhanced_r', 0),
            avg_metrics.get('pesq_enhanced_l', 0) - avg_metrics.get('pesq_noisy_l', 0),
            avg_metrics.get('pesq_enhanced_r', 0) - avg_metrics.get('pesq_noisy_r', 0),
            
            avg_metrics.get('snr_noisy_l', 0), avg_metrics.get('snr_noisy_r', 0),
            avg_metrics.get('snr_enhanced_l', 0), avg_metrics.get('snr_enhanced_r', 0),
            avg_metrics.get('snr_enhanced_l', 0) - avg_metrics.get('snr_noisy_l', 0),
            avg_metrics.get('snr_enhanced_r', 0) - avg_metrics.get('snr_noisy_r', 0),
            
            avg_metrics.get('ild_error', 0), avg_metrics.get('ipd_error', 0)
        ]
    }
    
    summary_df = pd.DataFrame(summary_metrics)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    
    print(f"\nEvaluation completed. Results saved to {output_dir}")
    
    return metrics, avg_metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate binaural speech enhancement model")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to the config directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--noisy_data", type=str, default=None, help="Path to noisy test dataset")
    parser.add_argument("--clean_data", type=str, default=None, help="Path to clean test dataset")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    evaluate_model(
        config_path=args.config_path,
        model_checkpoint_path=args.checkpoint,
        noisy_dataset_path=args.noisy_data,
        clean_dataset_path=args.clean_data,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )