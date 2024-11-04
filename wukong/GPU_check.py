import torch

if torch.cuda.is_available():
    print("CUDA is available. Number of devices:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print(torch.__version__)
    print(torch.version.cuda)
else:
    print("CUDA is not available. Running on CPU.")
