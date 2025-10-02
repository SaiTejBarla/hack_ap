import torch

print("PyTorch version:", torch.__version__)

# Check CUDA (NVIDIA GPU)
print("CUDA available:", torch.cuda.is_available())

# Check MPS (Apple GPUs)
print("MPS available:", torch.backends.mps.is_available())

# Check oneAPI (Intel GPUs)
print("oneAPI available:", hasattr(torch.backends, "oneapi") and torch.backends.oneapi.is_available())

# Print device name if available
if torch.cuda.is_available():
    print("CUDA GPU:", torch.cuda.get_device_name(0))
elif hasattr(torch.backends, "oneapi") and torch.backends.oneapi.is_available():
    print("Intel oneAPI GPU detected!")
else:
    print("No compatible GPU detected.")
