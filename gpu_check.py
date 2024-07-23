import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("PyTorch CUDA device count:", torch.cuda.device_count())
    print("PyTorch CUDA current device:", torch.cuda.current_device())
    print("PyTorch CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))