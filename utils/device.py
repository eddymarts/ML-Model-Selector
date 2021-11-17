import torch
from multiprocessing import cpu_count

def get_device():
    device = "cpu"
    num_cpu = cpu_count() -1
    if torch.cuda.is_available():
        device = "cuda:0"
        num_gpu = 8
        # num_gpu = torch.cuda.device_count() - 1
    else:
        num_gpu = 0

    return device, num_cpu, num_gpu