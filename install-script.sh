#!/bin/bash
# Script to set up the environment and run the evaluation for BCCTN

set -e  # Exit on error

# Define colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== BCCTN Model Evaluation Setup ===${NC}"

# Check Python version
python_version=$(python3 --version | cut -d ' ' -f 2)
echo -e "${YELLOW}Using Python version:${NC} $python_version"

# Create directory structure if needed
echo -e "${YELLOW}Creating directory structure...${NC}"
mkdir -p DCNN/utils

# Fix the base_trainer.py file
echo -e "${YELLOW}Installing fixed base_trainer.py...${NC}"
cp fixed_base_trainer.py DCNN/utils/base_trainer.py

# Check if the checkpoint exists
CHECKPOINT_PATH="/home/R12K41024/Research/BCCTN/DCNN/Checkpoints/Trained_model.ckpt"
if [ -f "$CHECKPOINT_PATH" ]; then
    echo -e "${GREEN}Checkpoint found:${NC} $CHECKPOINT_PATH"
else
    echo -e "${RED}ERROR: Checkpoint not found at:${NC} $CHECKPOINT_PATH"
    echo -e "${YELLOW}Please check the checkpoint path or provide a valid path:${NC}"
    echo "Usage: ./setup_and_evaluate.sh [--checkpoint /path/to/checkpoint.ckpt]"
    exit 1
fi

# Make output directory
mkdir -p evaluation_results

# Optional: Install required Python packages
echo -e "${YELLOW}Installing required packages...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch_lightning librosa matplotlib scipy hydra-core tqdm soundfile

# Run the evaluation
echo -e "${GREEN}Running model evaluation...${NC}"
python3 evaluate_model.py --checkpoint $CHECKPOINT_PATH --output_dir evaluation_results

echo -e "${GREEN}Evaluation complete! Check evaluation_results directory for outputs.${NC}"
