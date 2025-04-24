#!/bin/bash

# run_evaluation.sh
# Script to evaluate the BCCTN model across different SNR levels
# Usage: ./run_evaluation.sh <checkpoint_path> <data_directory>

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <checkpoint_path> <data_directory> [output_directory]"
    echo "Example: $0 ./DCNN/Checkpoints/Trained_model.ckpt ./Dataset evaluation_results"
    exit 1
fi

CHECKPOINT_PATH="$1"
DATA_DIR="$2"
OUTPUT_DIR="${3:-evaluation_results_$(date +%Y%m%d_%H%M%S)}"
CONFIG_PATH="./config"
NUM_SAMPLES=50  # Number of samples per SNR level
SNR_LEVELS=(-6 -3 0 3 6 9 12 15)  # SNR levels to evaluate

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    exit 1
fi

# Look for clean and noisy subdirectories
CLEAN_DIR=""
NOISY_DIR=""

# Try to find the clean and noisy directories
for dir in "$DATA_DIR" "$DATA_DIR"/* ; do
    base_name=$(basename "$dir")
    if [[ "$base_name" == "clean" || "$base_name" == "clean_testset_1f" ]]; then
        CLEAN_DIR="$dir"
        echo "Found clean directory: $CLEAN_DIR"
    elif [[ "$base_name" == "noisy" || "$base_name" == "noisy_testset_1f" ]]; then
        NOISY_DIR="$dir"
        echo "Found noisy directory: $NOISY_DIR"
    fi
done

# If we haven't found the directories yet, look deeper
if [ -z "$CLEAN_DIR" ] || [ -z "$NOISY_DIR" ]; then
    for dir in "$DATA_DIR"/*/* ; do
        if [ -d "$dir" ]; then
            base_name=$(basename "$dir")
            if [[ "$base_name" == "clean" || "$base_name" == "clean_testset_1f" ]]; then
                CLEAN_DIR="$dir"
                echo "Found clean directory: $CLEAN_DIR"
            elif [[ "$base_name" == "noisy" || "$base_name" == "noisy_testset_1f" ]]; then
                NOISY_DIR="$dir"
                echo "Found noisy directory: $NOISY_DIR"
            fi
        fi
    done
fi

# Final check for directories
if [ -z "$CLEAN_DIR" ]; then
    echo "Error: Clean data directory not found in $DATA_DIR"
    echo "Please specify the clean directory explicitly with --clean_dir"
    exit 1
fi

if [ -z "$NOISY_DIR" ]; then
    echo "Error: Noisy data directory not found in $DATA_DIR"
    echo "Please specify the noisy directory explicitly with --noisy_dir"
    exit 1
fi

echo "======================================"
echo "Running BCCTN Evaluation"
echo "======================================"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Clean data: $CLEAN_DIR"
echo "Noisy data: $NOISY_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "SNR levels: ${SNR_LEVELS[*]}"
echo "Samples per SNR: $NUM_SAMPLES"
echo "======================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the evaluation script
python evaluate_model.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --data_root "$DATA_DIR" \
    --clean_dir "$CLEAN_DIR" \
    --noisy_dir "$NOISY_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --config_path "$CONFIG_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --snr_levels "${SNR_LEVELS[@]}" \
    --save_audio

# Check if evaluation completed successfully
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo "Results are in: $OUTPUT_DIR"
    echo "Summary report: $OUTPUT_DIR/summary_report.md"
else
    echo "Evaluation failed with error code $?"
fi