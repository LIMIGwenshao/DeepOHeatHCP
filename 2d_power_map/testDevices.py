import torch

# 检查可用的 GPU 数量
gpu_count = torch.cuda.device_count()
print(f"Number of GPUs available: {gpu_count}")

# 打印每个 GPU 的详细信息
for i in range(gpu_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory Allocated: {torch.cuda.memory_allocated(i)} bytes")
    print(f"  Memory Cached: {torch.cuda.memory_reserved(i)} bytes")
