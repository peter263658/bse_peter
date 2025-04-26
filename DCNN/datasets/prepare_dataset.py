#!/usr/bin/env python3
"""
Binaural Dataset Preparation Script for BCCTN Model

This script prepares the binaural dataset by:
1. Loading clean speech from VCTK
2. Downsampling to 16 kHz if needed
3. Spatializing it using HRIR database
4. Generating isotropic noise fields
5. Mixing the spatialized speech with noise at different SNRs
6. Splitting into train/validation/test sets
7. Saving the binaural pairs in the expected directories

Requirements:
- librosa
- numpy
- scipy
- tqdm
- soundfile
- pathlib
- random
"""

import os
import random
import numpy as np
import librosa
import soundfile as sf
import scipy.signal as signal
from pathlib import Path
from tqdm import tqdm
import scipy.io as sio
import math
import shutil
import argparse

# Default paths - update these to your local paths
DEFAULT_VCTK_PATH = "/home/R12K41024/Dataset/VCTK-Corpus/VCTK-Corpus"
DEFAULT_NOISEX_PATH = "/home/R12K41024/Dataset/Noises-master/NoiseX-92"
DEFAULT_HRIR_PATH = "/home/R12K41024/Dataset/HRIR_database_mat/hrir/anechoic" 
DEFAULT_OUTPUT_PATH = "./DATASET"  # Output directory for the dataset

# Constants
TARGET_SR = 16000  # 16 kHz target sampling rate
SEGMENT_DURATION = 2.0  # 2-second segments as mentioned in the paper
SNR_RANGE_TRAIN = (-7, 16)  # SNR range for training set in dB
SNR_RANGE_TEST = (-6, 15)   # SNR range for test set in dB
AZIMUTH_RANGE = (-90, 90)   # Frontal plane azimuth range in degrees

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare binaural dataset for BCCTN model')
    parser.add_argument('--vctk', default=DEFAULT_VCTK_PATH, help='Path to VCTK corpus')
    parser.add_argument('--noisex', default=DEFAULT_NOISEX_PATH, help='Path to NOISEX-92 dataset')
    parser.add_argument('--hrir', default=DEFAULT_HRIR_PATH, help='Path to HRIR database')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_PATH, help='Output directory for dataset')
    parser.add_argument('--n_train', type=int, default=16000, help='Number of training samples')
    parser.add_argument('--n_val', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--n_test', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--snr_levels', type=float, nargs='+', 
                        default=[-6, -3, 0, 3, 6, 9, 12, 15],
                        help='Specific SNR levels to use (in dB)')
    parser.add_argument('--snr_per_level', type=int, default=500,
                        help='Number of samples per SNR level')
    return parser.parse_args()

def get_hrir_database(hrir_path):
    """
    Load the HRIR database from a directory of individual MAT files
    Each MAT file contains HRIR for a specific azimuth, elevation, and distance
    """
    print(f"Loading HRIR database from {hrir_path}")
    
    if not os.path.isdir(hrir_path):
        raise ValueError(f"{hrir_path} is not a directory")
    
    # Get all .mat files in the directory
    mat_files = [os.path.join(hrir_path, f) for f in os.listdir(hrir_path) if f.endswith('.mat')]
    
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {hrir_path}")
    
    print(f"Found {len(mat_files)} HRIR .mat files")
    
    # Parse filenames to extract metadata
    # Format: anechoic_distcm_X_el_Y_az_Z.mat
    azimuths = []
    elevations = []
    distances = []
    hrirs_list = []
    
    for mat_file in mat_files:
        filename = os.path.basename(mat_file)
        parts = filename.split('_')
        
        # Extract distance, elevation, azimuth from filename
        if len(parts) < 7:
            print(f"Warning: Unexpected filename format for {filename}, skipping.")
            continue
            
        try:
            dist_cm = int(parts[2])
            el_deg = int(parts[4])
            az_deg = int(parts[6].split('.')[0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse metadata from {filename}: {e}")
            continue
        
        # Load the MAT file
        try:
            data = sio.loadmat(mat_file)
            
            # Find the HRIR data in the MAT file
            # According to HRIR_README, the first two channels are in-ear
            hrir_data = None
            
            # Try standard key names first
            for key in ['hrir', 'h_data', 'IR', 'impulse_response', 'h', 'data']:
                if key in data and isinstance(data[key], np.ndarray):
                    hrir_data = data[key]
                    break
            
            # If still not found, try to identify by array shape
            if hrir_data is None:
                for key, value in data.items():
                    if key.startswith('__'):  # Skip metadata
                        continue
                    if isinstance(value, np.ndarray) and (
                        value.shape[0] == 2 or  # [left,right,samples]
                        (len(value.shape) > 1 and value.shape[1] >= 2)  # [samples,channels]
                    ):
                        hrir_data = value
                        print(f"Found likely HRIR data in key '{key}' with shape {value.shape}")
                        break
            
            if hrir_data is None:
                print(f"Warning: Could not find HRIR data in {filename}, skipping")
                continue
                
            # Store metadata and HRIR
            distances.append(dist_cm)
            elevations.append(el_deg)
            azimuths.append(az_deg)
            hrirs_list.append(hrir_data)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not hrirs_list:
        raise ValueError("Failed to load any HRIR data")
    
    # Determine if the HRIRs are for left/right ears
    # Log the shape of each HRIR for debugging
    shapes = [h.shape for h in hrirs_list]
    print(f"Found HRIR shapes: {set(shapes)}")
    
    # Reorganize data for easier access
    hrir_db = {
        'hrirs': hrirs_list,
        'azimuths': np.array(azimuths),
        'elevations': np.array(elevations),
        'distances': np.array(distances),
        'shapes': shapes
    }
    
    return hrir_db

def get_hrir_database(hrir_path):
    """
    Load the HRIR database from a directory of individual MAT files
    Each MAT file contains HRIR for a specific azimuth, elevation, and distance
    """
    print(f"Loading HRIR database from {hrir_path}")
    
    if not os.path.isdir(hrir_path):
        raise ValueError(f"{hrir_path} is not a directory")
    
    # Get all .mat files in the directory
    mat_files = [os.path.join(hrir_path, f) for f in os.listdir(hrir_path) if f.endswith('.mat')]
    
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {hrir_path}")
    
    print(f"Found {len(mat_files)} HRIR .mat files")
    
    # Parse filenames to extract metadata
    # Format: anechoic_distcm_X_el_Y_az_Z.mat
    azimuths = []
    elevations = []
    distances = []
    hrirs_list = []
    
    for mat_file in mat_files:
        filename = os.path.basename(mat_file)
        parts = filename.split('_')
        
        # Extract distance, elevation, azimuth from filename
        if len(parts) < 7:
            print(f"Warning: Unexpected filename format for {filename}, skipping.")
            continue
            
        try:
            dist_cm = int(parts[2])
            el_deg = int(parts[4])
            az_deg = int(parts[6].split('.')[0])
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse metadata from {filename}: {e}")
            continue
        
        # Load the MAT file
        try:
            data = sio.loadmat(mat_file)
            
            # Find the HRIR data in the MAT file
            # According to HRIR_README, the first two channels are in-ear
            hrir_data = None
            
            # Try standard key names first
            for key in ['hrir', 'h_data', 'IR', 'impulse_response', 'h', 'data']:
                if key in data and isinstance(data[key], np.ndarray):
                    hrir_data = data[key]
                    break
            
            # If still not found, try to identify by array shape
            if hrir_data is None:
                for key, value in data.items():
                    if key.startswith('__'):  # Skip metadata
                        continue
                    if isinstance(value, np.ndarray) and (
                        value.shape[0] == 2 or  # [left,right,samples]
                        (len(value.shape) > 1 and value.shape[1] >= 2)  # [samples,channels]
                    ):
                        hrir_data = value
                        print(f"Found likely HRIR data in key '{key}' with shape {value.shape}")
                        break
            
            if hrir_data is None:
                print(f"Warning: Could not find HRIR data in {filename}, skipping")
                continue
                
            # Store metadata and HRIR
            distances.append(dist_cm)
            elevations.append(el_deg)
            azimuths.append(az_deg)
            hrirs_list.append(hrir_data)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not hrirs_list:
        raise ValueError("Failed to load any HRIR data")
    
    # Determine if the HRIRs are for left/right ears
    # Log the shape of each HRIR for debugging
    shapes = [h.shape for h in hrirs_list]
    print(f"Found HRIR shapes: {set(shapes)}")
    
    # Reorganize data for easier access
    hrir_db = {
        'hrirs': hrirs_list,
        'azimuths': np.array(azimuths),
        'elevations': np.array(elevations),
        'distances': np.array(distances),
        'shapes': shapes
    }
    
    return hrir_db

def test_convolution(speech_file, hrir_db):
    """Test a single convolution to make sure everything works"""
    speech, sr = librosa.load(speech_file, sr=TARGET_SR, mono=True)
    speech = speech[:1000]  # Just use a short segment
    
    # Try spatializing at 0 degrees azimuth
    try:
        binaural_speech = spatialize_speech(speech, hrir_db, 0, 0, 80)
        print(f"Convolution test successful: input shape {speech.shape}, output shape {binaural_speech.shape}")
        return True
    except Exception as e:
        print(f"Convolution test failed: {e}")
        return False

def get_closest_hrir(hrir_db, target_azimuth, target_elevation=0, target_distance=80):
    """
    Get the closest HRIR from the database for a given azimuth, elevation, and distance
    """
    azimuths = hrir_db['azimuths']
    elevations = hrir_db['elevations']
    distances = hrir_db['distances']
    hrirs = hrir_db['hrirs']
    
    # Calculate a combined distance metric for finding the closest HRIR
    # Handle wraparound for azimuth (e.g. -175° is close to 175°)
    azimuth_diffs = np.minimum(
        np.abs(azimuths - target_azimuth),
        360 - np.abs(azimuths - target_azimuth)
    )
    
    elevation_diffs = np.abs(elevations - target_elevation)
    distance_diffs = np.abs(distances - target_distance)
    
    # Normalize differences based on their ranges
    # Distance has more weight as per the HRIR_README
    if np.max(distance_diffs) > 0:
        distance_diffs = distance_diffs / np.max(distance_diffs) * 3  # Higher weight
    if np.max(elevation_diffs) > 0:
        elevation_diffs = elevation_diffs / np.max(elevation_diffs) * 2
    if np.max(azimuth_diffs) > 0:
        azimuth_diffs = azimuth_diffs / np.max(azimuth_diffs)
    
    # Combined distance metric
    combined_diffs = distance_diffs + elevation_diffs + azimuth_diffs
    
    # Find the index of the closest match
    closest_idx = np.argmin(combined_diffs)
    
    # Get the corresponding HRIR
    hrir = hrirs[closest_idx]
    
    # Determine left and right ear channels based on shape
    if len(hrir.shape) == 2:
        if hrir.shape[0] == 2:  # (2, samples) - channels first
            left_hrir = hrir[0]
            right_hrir = hrir[1]
        elif hrir.shape[1] >= 2:  # (samples, channels) - channels last
            left_hrir = hrir[:, 0] 
            right_hrir = hrir[:, 1]
        else:
            raise ValueError(f"Unexpected HRIR shape: {hrir.shape}")
    else:
        raise ValueError(f"HRIR does not have expected dimensionality: {hrir.shape}")
    
    return left_hrir, right_hrir


def spatialize_speech(speech, hrir_db, azimuth, elevation=0, distance=80, sr=TARGET_SR):
    """
    Spatialize monaural speech using HRIR at specified azimuth and elevation
    """
    # Get HRIR for the specified azimuth and elevation
    left_hrir, right_hrir = get_closest_hrir(hrir_db, azimuth, elevation, distance)
    
    # Make sure the HRIRs are 1D arrays
    left_hrir = left_hrir.flatten()
    right_hrir = right_hrir.flatten()
    
    # Ensure speech is also 1D
    speech = speech.flatten()
    
    # Apply HRIRs to speech with full convolution
    # According to HRIR_README, should use 'full' and then handle the result
    left_channel = signal.convolve(speech, left_hrir, mode='full')
    right_channel = signal.convolve(speech, right_hrir, mode='full')
    
    # Trim to match original speech length with proper alignment
    # HRIR causes delay, so we need to handle this carefully
    hrir_len = len(left_hrir)
    speech_len = len(speech)
    delay = hrir_len // 2  # Approximate delay introduced by HRIR
    
    # Trim convolution result accounting for delay
    left_channel = left_channel[delay:delay + speech_len]
    right_channel = right_channel[delay:delay + speech_len]
    
    # Normalize according to HRIR_README
    max_amp = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
    if max_amp > 0:
        left_channel = left_channel / max_amp
        right_channel = right_channel / max_amp
    
    # Stack channels
    binaural_speech = np.vstack((left_channel, right_channel))
    
    return binaural_speech


def generate_isotropic_noise(noise_samples, hrir_db, duration, sr=TARGET_SR):
    """
    Generate isotropic noise field by spatializing uncorrelated noise sources
    at uniformly spaced azimuths (every 5 degrees)
    """
    # Number of noise sources (every 5 degrees in azimuthal plane)
    n_sources = 72  # 360 / 5 = 72
    azimuths = np.linspace(0, 355, n_sources)
    
    # Initialize binaural noise
    n_samples = int(duration * sr)
    left_noise = np.zeros(n_samples)
    right_noise = np.zeros(n_samples)
    
    # For each azimuth, generate a spatialized noise
    for azimuth in azimuths:
        # Get random segment from noise samples
        if len(noise_samples) > n_samples:
            start_idx = random.randint(0, len(noise_samples) - n_samples)
            noise_segment = noise_samples[start_idx:start_idx + n_samples]
        else:
            # If noise file is too short, repeat it
            repeats = math.ceil(n_samples / len(noise_samples))
            noise_segment = np.tile(noise_samples, repeats)[:n_samples]
        
        # Get HRIR for this azimuth
        left_hrir, right_hrir = get_closest_hrir(hrir_db, azimuth)
        
        # Apply HRIR to noise (spatial convolution)
        left_spatialized = signal.convolve(noise_segment, left_hrir, mode='same')
        right_spatialized = signal.convolve(noise_segment, right_hrir, mode='same')
        
        # Add to the binaural noise
        left_noise += left_spatialized
        right_noise += right_spatialized
    
    # Normalize
    max_val = max(np.max(np.abs(left_noise)), np.max(np.abs(right_noise)))
    if max_val > 0:
        left_noise /= max_val
        right_noise /= max_val
    
    return np.vstack((left_noise, right_noise))

# def spatialize_speech(speech, hrir_db, azimuth, elevation=0, distance=80, sr=TARGET_SR):
    """
    Spatialize monaural speech using HRIR at specified azimuth and elevation
    """
    # Get HRIR for the specified azimuth and elevation
    left_hrir, right_hrir = get_closest_hrir(hrir_db, azimuth, elevation, distance)
    
    # Make sure the HRIRs are 1D arrays
    left_hrir = left_hrir.flatten()
    right_hrir = right_hrir.flatten()
    
    # Ensure speech is also 1D
    speech = speech.flatten()
    
    # Apply HRIRs to speech (spatial convolution)
    # Use mode='full' and truncate to the length of the speech
    left_channel = signal.convolve(speech, left_hrir, mode='same')
    right_channel = signal.convolve(speech, right_hrir, mode='same')
    
    # Stack channels
    binaural_speech = np.vstack((left_channel, right_channel))
    
    return binaural_speech

def adjust_snr(speech, noise, target_snr):
    """
    Adjust the level of the noise to achieve the target SNR
    """
    # Calculate signal power
    speech_power = np.mean(speech[0]**2) + np.mean(speech[1]**2)
    
    # Calculate noise power
    noise_power = np.mean(noise[0]**2) + np.mean(noise[1]**2)
    
    # Calculate gain to apply to the noise
    gain = np.sqrt(speech_power / (noise_power * 10**(target_snr/10)))
    
    # Apply gain to the noise
    adjusted_noise = noise * gain
    
    return adjusted_noise

def create_directories(output_path, snr_levels):
    """Create the necessary directory structure for the dataset with SNR-specific folders"""
    dirs = {}
    
    # Create base directories
    for split in ['train', 'val', 'test']:
        for condition in ['clean', 'noisy']:
            base_dir = os.path.join(output_path, f"{condition}_{split}set_1f")
            os.makedirs(base_dir, exist_ok=True)
            dirs[f"{condition}_{split}"] = base_dir
            
            # Create SNR-specific directories
            for snr in snr_levels:
                snr_dir = os.path.join(base_dir, f"SNR_{int(snr)}dB")
                os.makedirs(snr_dir, exist_ok=True)
                dirs[f"{condition}_{split}_snr{int(snr)}"] = snr_dir
    
    return dirs

def process_vctk_files(vctk_path):
    """
    Process VCTK corpus to get a list of all audio files
    Split into train, validation, and test sets as per paper
    """
    all_files = []
    
    # Walk through the VCTK directory
    for root, _, files in os.walk(vctk_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                all_files.append(os.path.join(root, file))
    
    # Sort files to ensure reproducibility
    all_files.sort()
    
    # According to the paper, they used VCTK for training and validation
    # and reserved TIMIT for additional testing
    # Here we'll split VCTK into train and validation, and a small part for testing
    
    random.seed(42)  # For reproducibility
    random.shuffle(all_files)
    
    # Calculate the number of files for each set
    total_files = len(all_files)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1
    
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    # Split files
    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]
    
    print(f"Total VCTK files: {total_files}")
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files

def process_noise_files(noisex_path):
    """Process NoiseX-92 dataset to get a list of noise files"""
    noise_files = []
    
    # Walk through the NoiseX directory
    for root, _, files in os.walk(noisex_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                noise_files.append(os.path.join(root, file))
    
    # Filter by noise types mentioned in the paper
    # "White Gaussian Noise (WGN), Speech Shaped Noise (SSN), factory noise, and office noise"
    # This filtering might need to be adjusted based on your NoiseX-92 structure
    relevant_noise_types = ['white', 'ssn', 'factory', 'office', 'babble']
    
    filtered_noise_files = []
    for file in noise_files:
        if any(noise_type in file.lower() for noise_type in relevant_noise_types):
            filtered_noise_files.append(file)
    
    print(f"Found {len(filtered_noise_files)} relevant noise files")
    return filtered_noise_files

def process_split_with_fixed_snr(split_name, speech_files, noise_files, hrir_db, dirs, n_samples, target_snr):
    """Process a data split with a fixed SNR level"""
    print(f"Processing {split_name} split at SNR={target_snr}dB - generating {n_samples} samples")
    
    if split_name == 'train':
        clean_dir = dirs[f'clean_train_snr{int(target_snr)}']
        noisy_dir = dirs[f'noisy_train_snr{int(target_snr)}']
    elif split_name == 'validation':
        clean_dir = dirs[f'clean_val_snr{int(target_snr)}']
        noisy_dir = dirs[f'noisy_val_snr{int(target_snr)}']
    else:  # test
        clean_dir = dirs[f'clean_test_snr{int(target_snr)}']
        noisy_dir = dirs[f'noisy_test_snr{int(target_snr)}']
    
    # Load and cache noise files
    noise_data = {}
    for noise_file in noise_files:
        try:
            audio, sr = librosa.load(noise_file, sr=TARGET_SR, mono=True)
            noise_data[noise_file] = audio
        except Exception as e:
            print(f"Error loading noise file {noise_file}: {e}")
    
    # Generate samples
    sample_index = 0
    progress_bar = tqdm(total=n_samples, desc=f"{split_name} samples at SNR={target_snr}dB")
    
    while sample_index < n_samples:
        # Select random speech file
        speech_file = random.choice(speech_files)
        
        try:
            # Load and preprocess speech
            speech, sr = librosa.load(speech_file, sr=TARGET_SR, mono=True)
            
            # Make sure it's long enough
            min_samples = int(SEGMENT_DURATION * TARGET_SR)
            if len(speech) < min_samples:
                continue
            
            # Select random segment if file is longer than needed
            if len(speech) > min_samples:
                start = random.randint(0, len(speech) - min_samples)
                speech = speech[start:start + min_samples]
            
            # Normalize speech
            speech = speech / np.max(np.abs(speech))
            
            # Generate random azimuth in frontal plane
            azimuth = random.uniform(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])
            
            # Generate random elevation (0 for this implementation)
            elevation = 0
            
            # Select random distance (80cm or 300cm as per paper)
            distance = random.choice([80, 300])
            
            # Spatialize speech
            binaural_speech = spatialize_speech(speech, hrir_db, azimuth, elevation)
            
            # Select random noise file
            noise_file = random.choice(list(noise_data.keys()))
            noise_samples = noise_data[noise_file]
            
            # Generate isotropic noise
            binaural_noise = generate_isotropic_noise(noise_samples, hrir_db, SEGMENT_DURATION)
            
            # Use the fixed target SNR instead of a random one
            # Adjust noise level to achieve target SNR
            adjusted_noise = adjust_snr(binaural_speech, binaural_noise, target_snr)
            
            # Mix speech and noise
            noisy_binaural = binaural_speech + adjusted_noise
            
            # Normalize both clean and noisy signals to avoid clipping
            clean_max = np.max(np.abs(binaural_speech))
            if clean_max > 0:
                binaural_speech = binaural_speech / clean_max * 0.9
            
            noisy_max = np.max(np.abs(noisy_binaural))
            if noisy_max > 0:
                noisy_binaural = noisy_binaural / noisy_max * 0.9
            
            # Generate unique filename
            speaker_id = os.path.basename(os.path.dirname(speech_file))
            filename = f"{speaker_id}_{sample_index:06d}_az{int(azimuth):+04d}_el{int(elevation):+03d}_d{int(distance):03d}_snr{int(target_snr):+03d}.wav"
            
            # Save clean and noisy binaural files
            clean_path = os.path.join(clean_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)
            
            sf.write(clean_path, binaural_speech.T, TARGET_SR)
            sf.write(noisy_path, noisy_binaural.T, TARGET_SR)
            
            sample_index += 1
            progress_bar.update(1)
            
        except Exception as e:
            print(f"Error processing {speech_file}: {e}")
    
    progress_bar.close()

def create_binaural_dataset(args):
    """Main function to create the binaural dataset with specific SNR levels"""
    
    # Load HRIR database
    hrir_db = get_hrir_database(args.hrir)
    
    # Get speech and noise files
    train_files, val_files, test_files = process_vctk_files(args.vctk)
    noise_files = process_noise_files(args.noisex)
    
    # Test convolution
    if train_files:
        success = test_convolution(train_files[0], hrir_db)
        if not success:
            print("Aborting due to convolution test failure")
            return
    
    # Create directory structure with SNR levels
    dirs = create_directories(args.output, args.snr_levels)
    
    # Process each split for each SNR level
    for split_name, files, n_base in [
        ('train', train_files, args.n_train), 
        ('validation', val_files, args.n_val), 
        ('test', test_files, args.n_test)
    ]:
        # Calculate samples per SNR level
        n_per_snr = args.snr_per_level if hasattr(args, 'snr_per_level') else n_base // len(args.snr_levels)
        
        print(f"Processing {split_name} split - {n_per_snr} samples per SNR level")
        
        # Process each SNR level separately
        for snr in args.snr_levels:
            process_split_with_fixed_snr(
                split_name, 
                files, 
                noise_files, 
                hrir_db, 
                dirs, 
                n_per_snr, 
                snr
            )
    
    print("Dataset preparation complete!")



def process_split(split_name, speech_files, noise_files, hrir_db, dirs, n_samples, snr_range):
    """Process a data split (train, validation, or test)"""
    print(f"Processing {split_name} split - generating {n_samples} samples")
    
    if split_name == 'train':
        clean_dir = dirs['clean_train']
        noisy_dir = dirs['noisy_train']
    elif split_name == 'validation':
        clean_dir = dirs['clean_val']
        noisy_dir = dirs['noisy_val']
    else:  # test
        clean_dir = dirs['clean_test']
        noisy_dir = dirs['noisy_test']
    
    # Load and cache noise files
    noise_data = {}
    for noise_file in noise_files:
        try:
            audio, sr = librosa.load(noise_file, sr=TARGET_SR, mono=True)
            noise_data[noise_file] = audio
        except Exception as e:
            print(f"Error loading noise file {noise_file}: {e}")
    
    # Generate samples
    sample_index = 0
    progress_bar = tqdm(total=n_samples, desc=f"{split_name} samples")
    
    while sample_index < n_samples:
        # Select random speech file
        speech_file = random.choice(speech_files)
        
        try:
            # Load and preprocess speech
            speech, sr = librosa.load(speech_file, sr=TARGET_SR, mono=True)
            
            # Make sure it's long enough
            min_samples = int(SEGMENT_DURATION * TARGET_SR)
            if len(speech) < min_samples:
                continue
            
            # Select random segment if file is longer than needed
            if len(speech) > min_samples:
                start = random.randint(0, len(speech) - min_samples)
                speech = speech[start:start + min_samples]
            
            # Normalize speech
            speech = speech / np.max(np.abs(speech))
            
            # Generate random azimuth in frontal plane
            azimuth = random.uniform(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1])
            
            # Generate random elevation (0 for this implementation)
            elevation = 0
            
            # Select random distance (80cm or 300cm as per paper)
            distance = random.choice([80, 300])
            
            # Spatialize speech
            binaural_speech = spatialize_speech(speech, hrir_db, azimuth, elevation)
            
            # Select random noise file
            noise_file = random.choice(list(noise_data.keys()))
            noise_samples = noise_data[noise_file]
            
            # Generate isotropic noise
            binaural_noise = generate_isotropic_noise(noise_samples, hrir_db, SEGMENT_DURATION)
            
            # Select random SNR
            target_snr = random.uniform(snr_range[0], snr_range[1])
            
            # Adjust noise level to achieve target SNR
            adjusted_noise = adjust_snr(binaural_speech, binaural_noise, target_snr)
            
            # Mix speech and noise
            noisy_binaural = binaural_speech + adjusted_noise
            
            # Normalize both clean and noisy signals to avoid clipping
            clean_max = np.max(np.abs(binaural_speech))
            if clean_max > 0:
                binaural_speech = binaural_speech / clean_max * 0.9
            
            noisy_max = np.max(np.abs(noisy_binaural))
            if noisy_max > 0:
                noisy_binaural = noisy_binaural / noisy_max * 0.9
            
            # Generate unique filename
            speaker_id = os.path.basename(os.path.dirname(speech_file))
            filename = f"{speaker_id}_{sample_index:06d}_az{int(azimuth):+04d}_el{int(elevation):+03d}_d{int(distance):03d}_snr{int(target_snr):+03d}.wav"
            
            # Save clean and noisy binaural files
            clean_path = os.path.join(clean_dir, filename)
            noisy_path = os.path.join(noisy_dir, filename)
            
            sf.write(clean_path, binaural_speech.T, TARGET_SR)
            sf.write(noisy_path, noisy_binaural.T, TARGET_SR)
            
            sample_index += 1
            progress_bar.update(1)
            
        except Exception as e:
            print(f"Error processing {speech_file}: {e}")
    
    progress_bar.close()

if __name__ == "__main__":
    args = parse_arguments()
    create_binaural_dataset(args)
