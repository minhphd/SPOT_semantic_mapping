import torch

# Check if CUDA is available (meaning a GPU is detected)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

# If CUDA is available, print the number of GPUs and the name of the first one
if cuda_available:
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

