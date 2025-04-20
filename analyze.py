#!/usr/bin/env python
# coding: utf-8

import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from tqdm import tqdm
import sys
import os
from pathlib import Path

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

from DCNN.trainer import DCNNLightningModule
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.feature_extractors import Stft, IStft
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility

def compute_ild(s1, s2, eps=1e-6):
    """Compute Interaural Level Difference in dB"""
    l1 = 20 * np.log10(np.abs(s1) + eps)
    l2 = 20 * np.log10(np.abs(s2) + eps)
    return l1 - l2

def compute_ipd(s1, s2, eps=1e-6):
    """Compute Interaural Phase Difference in radians"""
    return np.angle(s1 / (s2 + eps))

def analyze_data(config_path="./config", 
                model_checkpoint_path="DCNN/Checkpoints/Trained_model.ckpt",
                noisy_dataset_path=None, 
                clean_dataset_path=None,
                output_dir="analysis_results", 
                num_samples=5):
    """
    Analyze the performance of the trained model
    
    Args:
        config_path: Path to the config directory
        model_checkpoint_path: Path to the saved model checkpoint
        noisy_dataset_path: Path to noisy test dataset, if None uses config value
        clean_dataset_path: Path to clean test dataset, if None uses config value 
        output_dir: Directory to save visualizations and results
        num_samples: Number of samples to analyze
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
    
    # Check if paths exist
    for path in [noisy_dataset_path, clean_dataset_path]:
        if not os.path.exists(path):
            print(f"Error: Path {path} does not exist.")
            return
    
    print(f"Loading model from {model_checkpoint_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load checkpoint
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    
    print(f"Model loaded successfully. Starting analysis on {num_samples} samples...")
    
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
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=2
    )
    
    # Initialize metrics storage
    metrics = {
        'stoi_noisy': [],
        'stoi_enhanced': [],
        'snr_noisy': [],
        'snr_enhanced': []
    }
    
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
            
            # Calculate STOI
            stoi_noisy_l = stoi_metric(noisy_samples[0, 0], clean_samples[0, 0]).item()
            stoi_noisy_r = stoi_metric(noisy_samples[0, 1], clean_samples[0, 1]).item()
            stoi_enhanced_l = stoi_metric(model_output[0, 0], clean_samples[0, 0]).item()
            stoi_enhanced_r = stoi_metric(model_output[0, 1], clean_samples[0, 1]).item()
            
            metrics['stoi_noisy'].append((stoi_noisy_l + stoi_noisy_r) / 2)
            metrics['stoi_enhanced'].append((stoi_enhanced_l + stoi_enhanced_r) / 2)
            
            # Calculate SNR
            def calculate_snr(clean, processed):
                noise = processed - clean
                return 10 * torch.log10(torch.sum(clean**2) / (torch.sum(noise**2) + 1e-10))
            
            snr_noisy_l = calculate_snr(clean_samples[0, 0], noisy_samples[0, 0]).item()
            snr_noisy_r = calculate_snr(clean_samples[0, 1], noisy_samples[0, 1]).item()
            snr_enhanced_l = calculate_snr(clean_samples[0, 0], model_output[0, 0]).item()
            snr_enhanced_r = calculate_snr(clean_samples[0, 1], model_output[0, 1]).item()
            
            metrics['snr_noisy'].append((snr_noisy_l + snr_noisy_r) / 2)
            metrics['snr_enhanced'].append((snr_enhanced_l + snr_enhanced_r) / 2)
            
            # Convert to numpy for visualization
            noisy_np = noisy_samples[0].numpy()
            clean_np = clean_samples[0].numpy()
            enhanced_np = model_output[0].numpy()
            
            # Compute STFT for visualization
            noisy_stft_l = stft(noisy_samples[0, 0]).numpy()
            noisy_stft_r = stft(noisy_samples[0, 1]).numpy()
            enhanced_stft_l = stft(model_output[0, 0]).numpy()
            enhanced_stft_r = stft(model_output[0, 1]).numpy()
            clean_stft_l = stft(clean_samples[0, 0]).numpy()
            clean_stft_r = stft(clean_samples[0, 1]).numpy()
            
            # Compute interaural cues
            ild_noisy = compute_ild(noisy_stft_l, noisy_stft_r)
            ild_enhanced = compute_ild(enhanced_stft_l, enhanced_stft_r)
            ild_clean = compute_ild(clean_stft_l, clean_stft_r)
            
            ipd_noisy = compute_ipd(noisy_stft_l, noisy_stft_r)
            ipd_enhanced = compute_ipd(enhanced_stft_l, enhanced_stft_r)
            ipd_clean = compute_ipd(clean_stft_l, clean_stft_r)
            
            # Save audio files
            save_dir = os.path.join(output_dir, f"sample_{i}")
            os.makedirs(save_dir, exist_ok=True)
            
            # Print metrics
            print(f"\nSample {i} Metrics:")
            print(f"STOI: Noisy={metrics['stoi_noisy'][-1]:.3f}, Enhanced={metrics['stoi_enhanced'][-1]:.3f}")
            print(f"SNR: Noisy={metrics['snr_noisy'][-1]:.2f} dB, Enhanced={metrics['snr_enhanced'][-1]:.2f} dB")
            print(f"SNR Improvement: {metrics['snr_enhanced'][-1] - metrics['snr_noisy'][-1]:.2f} dB")
            
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
            plt.subplot(2, 1, 1)
            plt.plot(np.mean(ild_clean, axis=1), label='Clean')
            plt.plot(np.mean(ild_noisy, axis=1), label='Noisy')
            plt.plot(np.mean(ild_enhanced, axis=1), label='Enhanced')
            plt.xlabel('Frequency Bin')
            plt.ylabel('ILD (dB)')
            plt.title('Interaural Level Difference (ILD)')
            plt.legend()
            plt.grid(True)
            
            # IPD visualization
            plt.subplot(2, 1, 2)
            plt.plot(np.mean(ipd_clean, axis=1), label='Clean')
            plt.plot(np.mean(ipd_noisy, axis=1), label='Noisy')
            plt.plot(np.mean(ipd_enhanced, axis=1), label='Enhanced')
            plt.xlabel('Frequency Bin')
            plt.ylabel('IPD (rad)')
            plt.title('Interaural Phase Difference (IPD)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "interaural_cues.png"))
            plt.close()
            
            # If running in Jupyter notebook, display audio
            try:
                if 'ipykernel' in sys.modules:
                    print("Noisy Left:")
                    display(ipd.Audio(noisy_np[0], rate=SR))
                    print("Enhanced Left:")
                    display(ipd.Audio(enhanced_np[0], rate=SR))
                    print("Clean Left:")
                    display(ipd.Audio(clean_np[0], rate=SR))
                    
                    print("Noisy Right:")
                    display(ipd.Audio(noisy_np[1], rate=SR))
                    print("Enhanced Right:")
                    display(ipd.Audio(enhanced_np[1], rate=SR))
                    print("Clean Right:")
                    display(ipd.Audio(clean_np[1], rate=SR))
            except:
                pass
    
    # Print average metrics
    print("\nAverage Metrics:")
    print(f"STOI: Noisy={np.mean(metrics['stoi_noisy']):.3f}, Enhanced={np.mean(metrics['stoi_enhanced']):.3f}")
    print(f"SNR: Noisy={np.mean(metrics['snr_noisy']):.2f} dB, Enhanced={np.mean(metrics['snr_enhanced']):.2f} dB")
    print(f"Average SNR Improvement: {np.mean(np.array(metrics['snr_enhanced']) - np.array(metrics['snr_noisy'])):.2f} dB")
    
    # Create summary visualization
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(['Noisy', 'Enhanced'], [np.mean(metrics['stoi_noisy']), np.mean(metrics['stoi_enhanced'])])
    plt.ylabel('STOI')
    plt.title('Speech Intelligibility')
    plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    plt.bar(['Noisy', 'Enhanced'], [np.mean(metrics['snr_noisy']), np.mean(metrics['snr_enhanced'])])
    plt.ylabel('SNR (dB)')
    plt.title('Signal-to-Noise Ratio')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_metrics.png"))
    plt.close()
    
    # Save metrics to file
    import pandas as pd
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    print(f"\nAnalysis completed. Results saved to {output_dir}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze performance of binaural speech enhancement model")
    parser.add_argument("--config_path", type=str, default="./config", help="Path to the config directory")
    parser.add_argument("--checkpoint", type=str, default="DCNN/Checkpoints/Trained_model.ckpt", help="Path to model checkpoint")
    parser.add_argument("--noisy_data", type=str, default=None, help="Path to noisy test dataset")
    parser.add_argument("--clean_data", type=str, default=None, help="Path to clean test dataset")
    parser.add_argument("--output_dir", type=str, default="analysis_results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    analyze_data(
        config_path=args.config_path,
        model_checkpoint_path=args.checkpoint,
        noisy_dataset_path=args.noisy_data,
        clean_dataset_path=args.clean_data,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )