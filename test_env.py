import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check number of available GPUs
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Print GPU information
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
