"""Requirements for a new machine's env/: gpu-a240's exact versions, except the GPU-specific torch stack.

    python merge_reqs.py <gpu-a240_env.txt> <torch_reqs.txt> <out_requirements.txt> <out_deviations.txt>

- dropped: torch, torchvision, triton, nvidia-*-cu12/cu13 libraries (installed for the local GPU instead),
  causal-conv1d and mamba-ssm (built from repos/SegMamba by tools/build_segmamba.sh)
- torch's own requirements (from the installed torch's metadata): gpu-a240's pinned version is kept when it
  satisfies torch's specifier, otherwise torch's requirement wins; every such case is written to the deviations file.
Runs with any Python that has pip (uses pip's vendored `packaging`).
"""
import re
import sys

try:
    from packaging.requirements import Requirement
    from packaging.version import Version
except ImportError:
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.version import Version

a240_file, torch_file, out_file, dev_file = sys.argv[1:5]
TARGET = {"python_version": "3.11", "python_full_version": "3.11.15",   # env/'s interpreter and the servers'
          "platform_system": "Linux", "platform_machine": "x86_64", "sys_platform": "linux"}   # platform, not this one
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
GPU_STACK = re.compile(r"^(torch|torchvision|torchaudio|triton|nvidia-.*-cu1[0-9])$")
BUILT_LOCALLY = {"causal-conv1d", "mamba-ssm"}

a240 = {}                       # name -> original line
for line in open(a240_file):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    name = norm(re.split(r"[=<>!~ @\[;]", line, maxsplit=1)[0])
    a240[name] = line

torch_reqs = [Requirement(l.strip()) for l in open(torch_file) if l.strip()]
out, dev = [], []
for name, line in a240.items():
    if GPU_STACK.match(name) or name in BUILT_LOCALLY:
        dev.append(f"replaced  {line}  (GPU-specific / built locally)")
        continue
    reqs = [r for r in torch_reqs if norm(r.name) == name and (r.marker is None or r.marker.evaluate(TARGET))]
    if reqs and "==" in line and " @ " not in line:
        v = Version(line.split("==", 1)[1])
        bad = [r for r in reqs if not r.specifier.contains(v, prereleases=True)]
        if bad:
            out.append(f"{name}{bad[0].specifier}")
            dev.append(f"changed   {line} -> {name}{bad[0].specifier}  (required by the local torch)")
            continue
    out.append(line)
for r in torch_reqs:   # torch's own deps that are GPU-specific (NVIDIA libraries, triton) or new to gpu-a240's list
    n = norm(r.name)
    if n != "torch" and (n not in a240 or GPU_STACK.match(n)) and (r.marker is None or r.marker.evaluate(TARGET)):
        out.append(f"{r.name}{r.specifier}")
        dev.append(f"added     {r.name}{r.specifier}  (required by the local torch)")
open(out_file, "w").write("\n".join(out) + "\n")
open(dev_file, "w").write("\n".join(dev) + "\n")
print(f"{len(out)} requirements, {len(dev)} deviations from gpu-a240 -> {dev_file}")
