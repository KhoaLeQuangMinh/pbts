"""Memory/speed of SegMamba and Swin UNETR at the whole-head frame (one training step, AMP), fresh process each."""
import json, sys, time, torch
sys.path.insert(0, ".")
which, bs = sys.argv[1], int(sys.argv[2])
Z, Y, X = 160, 224, 192
if which == "segmamba":
    from models.segmamba.trainer import build_segmamba
    net = build_segmamba(4, 5, (Z, Y, X))
else:
    from monai.networks.nets import SwinUNETR
    net = SwinUNETR(img_size=(Z, Y, X), in_channels=4, out_channels=5, feature_size=48, use_checkpoint=True)
net = net.cuda().train(); opt = torch.optim.SGD(net.parameters(), lr=1e-3)
x = torch.randn(bs, 4, Z, Y, X, device="cuda"); torch.cuda.reset_peak_memory_stats(); ts = []
try:
    for i in range(4):
        torch.cuda.synchronize(); t = time.time()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = net(x).float().mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
        if i: ts.append(time.time() - t)
    r = {"peak_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1), "s_per_step": round(sum(ts) / len(ts), 2)}
except torch.cuda.OutOfMemoryError:
    r = {"peak_GB": "OOM"}
print(json.dumps({"model": which, "bs": bs, "gpu": torch.cuda.get_device_name(0), "params_M": round(sum(p.numel() for p in net.parameters()) / 1e6, 1), **r}))
