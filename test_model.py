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

from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize

from DCNN.trainer import DCNNLightningModule
from DCNN.datasets.base_dataset import BaseDataset
from DCNN.feature_extractors import Stft, IStft
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from pesq import pesq

def test_model(config_path="./config", model_checkpoint_path=None, 
               noisy_dataset_path=None, clean_dataset_path=None,
               output_dir=None, num_samples=5, random_seed=42):
    """
    Test the trained model on a test dataset and generate evaluation metrics
    
    Args:
        config_path: Path to the config directory
        model_checkpoint_path: Path to the saved model checkpoint
        noisy_dataset_path: Path to noisy test dataset
        clean_dataset_path: Path to clean test dataset
        output_dir: Directory to save enhanced audio files and visualizations
        num_samples: Number of samples to evaluate
        random_seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Initialize Hydra config
    GlobalHydra.instance().clear()
    initialize(config_path=config_path)
    config = compose("config")
    
    # If paths not provided, use from config
    if noisy_dataset_path is None:
        noisy_dataset_path = config.dataset.noisy_test_dataset_dir
    if clean_dataset_path is None:
        clean_dataset_path = config.dataset.target_test_dataset_dir
    
    # Default model checkpoint path
    if model_checkpoint_path is None:
        model_checkpoint_path = "DCNN/Checkpoints/Trained_model.ckpt"
    
    # Create output directory
    if output_dir is None:
        output_dir = "test_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    print(f"Loading model from {model_checkpoint_path}")
    model = DCNNLightningModule(config)
    model.eval()
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    checkpoint = torch.load(model_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.to(device)
    
    # Set up dataset and dataloader
    dataset = BaseDataset(noisy_dataset_path, clean_dataset_path, mono=False)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=2,
    )
    
    # Initialize metrics
    sr = 16000  # Sample rate
    stft = Stft(n_dft=512, hop_size=100, win_length=400)
    istft = IStft(n_dft=512, hop_size=100, win_length=400)
    stoi_metric = ShortTimeObjectiveIntelligibility(fs=sr)
    
    # Initialize results storage
    results = {
        'noisy_stoi': [],
        'enhanced_stoi': [],
        'noisy_pesq': [],
        'enhanced_pesq': [],
        'snr_improvement': []
    }
    
    # Process samples
    print(f"Processing {num_samples} samples...")
    
    for i, batch in enumerate(tqdm(dataloader, total=num_samples)):
        if i >= num_samples:
            break
        
        # Extract noisy and clean samples
        noisy_samples = batch[0].to(device)
        clean_samples = batch[1].to(device)
        
        # Get model output
        with torch.no_grad():
            model_output = model(noisy_samples)
        
        # Convert to numpy for evaluation
        noisy_np = noisy_samples[0].cpu().numpy()
        clean_np = clean_samples[0].cpu().numpy()
        enhanced_np = model_output[0].cpu().numpy()
        
        # Calculate STOI
        noisy_stoi_l = stoi_metric(torch.from_numpy(noisy_np[0]), torch.from_numpy(clean_np[0])).item()
        noisy_stoi_r = stoi_metric(torch.from_numpy(noisy_np[1]), torch.from_numpy(clean_np[1])).item()
        enhanced_stoi_l = stoi_metric(torch.from_numpy(enhanced_np[0]), torch.from_numpy(clean_np[0])).item()
        enhanced_stoi_r = stoi_metric(torch.from_numpy(enhanced_np[1]), torch.from_numpy(clean_np[1])).item()
        
        # Calculate PESQ when possible (may fail for very noisy signals)
        try:
            noisy_pesq_l = pesq(sr, clean_np[0], noisy_np[0], 'wb')
            noisy_pesq_r = pesq(sr, clean_np[1], noisy_np[1], 'wb')
            enhanced_pesq_l = pesq(sr, clean_np[0], enhanced_np[0], 'wb')
            enhanced_pesq_r = pesq(sr, clean_np[1], enhanced_np[1], 'wb')
        except Exception as e:
            print(f"PESQ calculation failed: {e}")
            noisy_pesq_l = noisy_pesq_r = enhanced_pesq_l = enhanced_pesq_r = 0
        
        # Calculate SNR
        def calculate_snr(clean, noisy):
            noise = noisy - clean
            return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-10))
        
        noisy_snr_l = calculate_snr(clean_np[0], noisy_np[0])
        noisy_snr_r = calculate_snr(clean_np[1], noisy_np[1])
        enhanced_snr_l = calculate_snr(clean_np[0], enhanced_np[0])
        enhanced_snr_r = calculate_snr(clean_np[1], enhanced_np[1])
        
        # Calculate improvement
        snr_improvement_l = enhanced_snr_l - noisy_snr_l
        snr_improvement_r = enhanced_snr_r - noisy_snr_r
        
        # Store results
        results['noisy_stoi'].append((noisy_stoi_l + noisy_stoi_r) / 2)
        results['enhanced_stoi'].append((enhanced_stoi_l + enhanced_stoi_r) / 2)
        results['noisy_pesq'].append((noisy_pesq_l + noisy_pesq_r) / 2)
        results['enhanced_pesq'].append((enhanced_pesq_l + enhanced_pesq_r) / 2)
        results['snr_improvement'].append((snr_improvement_l + snr_improvement_r) / 2)
        
        # Save audio files
        sf.write(f"{output_dir}/sample_{i}_noisy_L.wav", noisy_np[0], sr)
        sf.write(f"{output_dir}/sample_{i}_noisy_R.wav", noisy_np[1], sr)
        sf.write(f"{output_dir}/sample_{i}_enhanced_L.wav", enhanced_np[0], sr)
        sf.write(f"{output_dir}/sample_{i}_enhanced_R.wav", enhanced_np[1], sr)
        sf.write(f"{output_dir}/sample_{i}_clean_L.wav", clean_np[0], sr)
        sf.write(f"{output_dir}/sample_{i}_clean_R.wav", clean_np[1], sr)
        
        # Create spectrograms
        plt.figure(figsize=(15, 12))
        
        # Noisy spectrograms
        plt.subplot(3, 2, 1)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[0])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Noisy Left Channel')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 2, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_np[1])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Noisy Right Channel')
        plt.colorbar(format='%+2.0f dB')
        
        # Enhanced spectrograms
        plt.subplot(3, 2, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[0])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Enhanced Left Channel')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 2, 4)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(enhanced_np[1])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Enhanced Right Channel')
        plt.colorbar(format='%+2.0f dB')
        
        # Clean spectrograms
        plt.subplot(3, 2, 5)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[0])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Clean Left Channel')
        plt.colorbar(format='%+2.0f dB')
        
        plt.subplot(3, 2, 6)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(clean_np[1])), ref=np.max)
        librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr)
        plt.title('Clean Right Channel')
        plt.colorbar(format='%+2.0f dB')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sample_{i}_spectrograms.png")
        plt.close()
        
        # Print metrics for current sample
        print(f"\nSample {i} Metrics:")
        print(f"STOI: Noisy={results['noisy_stoi'][-1]:.3f}, Enhanced={results['enhanced_stoi'][-1]:.3f}")
        print(f"PESQ: Noisy={results['noisy_pesq'][-1]:.3f}, Enhanced={results['enhanced_pesq'][-1]:.3f}")
        print(f"SNR Improvement: {results['snr_improvement'][-1]:.2f} dB")
    
    # Calculate and print average metrics
    print("\nAverage Metrics:")
    print(f"STOI: Noisy={np.mean(results['noisy_stoi']):.3f}, Enhanced={np.mean(results['enhanced_stoi']):.3f}")
    print(f"PESQ: Noisy={np.mean(results['noisy_pesq']):.3f}, Enhanced={np.mean(results['enhanced_pesq']):.3f}")
    print(f"Average SNR Improvement: {np.mean(results['snr_improvement']):.2f} dB")
    
    # Save metrics to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/metrics.csv", index=False)
    
    # Create summary plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.bar(['Noisy', 'Enhanced'], [np.mean(results['noisy_stoi']), np.mean(results['enhanced_stoi'])])
    plt.title('Average STOI')
    plt.ylim(0, 1)
    
    plt.subplot(1, 3, 2)
    plt.bar(['Noisy', 'Enhanced'], [np.mean(results['noisy_pesq']), np.mean(results['enhanced_pesq'])])
    plt.title('Average PESQ')
    plt.ylim(1, 5)
    
    plt.subplot(1, 3, 3)
    plt.bar(['SNR Improvement'], [np.mean(results['snr_improvement'])])
    plt.title('Average SNR Improvement (dB)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/summary_metrics.png")
    
    print(f"Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test trained binaural speech enhancement model')
    parser.add_argument('--config_path', type=str, default='./config', help='Path to the config directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--noisy_data', type=str, default=None, help='Path to noisy test dataset')
    parser.add_argument('--clean_data', type=str, default=None, help='Path to clean test dataset')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    test_model(
        config_path=args.config_path,
        model_checkpoint_path=args.checkpoint,
        noisy_dataset_path=args.noisy_data,
        clean_dataset_path=args.clean_data,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        random_seed=args.seed
    )