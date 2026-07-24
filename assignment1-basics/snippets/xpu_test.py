import torch
print(f"PyTorch version: {torch.__version__}")
print(f"XPU compiled: {torch._C._xpu_getDeviceCount is not None}")
print(f"XPU available: {torch.xpu.is_available()}")
print(f"Device count: {torch.xpu.device_count()}")
print(f"Device name: {torch.xpu.get_device_name(0)}")

# Test tensor creation
x = torch.randn(4, 4, device="xpu", dtype=torch.bfloat16)
print(f"Test tensor: {x.size()} {x.device} dtype {x.dtype}")
