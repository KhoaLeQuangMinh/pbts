"""Does this machine's torch actually run on its GPU? (version, compiled architectures, one 3D conv step with AMP)."""
import torch

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version(),
      "| GPU available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0),
      "| torch compiled for", torch.cuda.get_arch_list())
try:
    x = torch.randn(2, 4, 64, 64, 64, device="cuda", requires_grad=True)
    conv = torch.nn.Conv3d(4, 32, 3, padding=1).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y = conv(x).float().mean()
    y.backward()
    torch.cuda.synchronize()
    print("GPU CHECK OK: conv3d forward+backward with AMP")
except Exception as e:
    print("GPU CHECK FAILED:", repr(e)[:400])
    raise SystemExit(1)
