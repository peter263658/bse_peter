#!/usr/bin/env python3
"""
This script runs the evaluation of the BCCTN model using the trained checkpoint.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run evaluation on BCCTN model")
    parser.add_argument("--checkpoint", type=str, 
                      default="/home/R12K41024/Research/BCCTN/DCNN/Checkpoints/Trained_model.ckpt",
                      help="Path to the model checkpoint")
    parser.add_argument("--config", type=str, default="./config",
                      help="Path to config directory")
    parser.add_argument("--num_samples", type=int, default=5,
                      help="Number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                      help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # First, check if the checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check if config directory exists
    if not os.path.exists(args.config):
        print(f"ERROR: Config directory not found: {args.config}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the evaluation
    cmd = (f"python3 evaluate_model.py "
           f"--checkpoint {args.checkpoint} "
           f"--config_path {args.config} "
           f"--num_samples {args.num_samples} "
           f"--output_dir {args.output_dir}")
    
    print(f"Running command: {cmd}")
    result = os.system(cmd)
    
    if result != 0:
        print("Evaluation failed with error code:", result)
        sys.exit(1)
    else:
        print(f"Evaluation completed successfully. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()