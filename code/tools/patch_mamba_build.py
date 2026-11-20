"""Let the SegMamba CUDA builds (causal-conv1d, mamba) target extra GPU architectures (idempotent).

Their setup.py hard-codes sm_70/sm_80/sm_90. The Blackwell machine's RTX PRO 5000 is sm_120. This adds, right after
the sm_90 entry, one gencode per architecture listed in PBTS_EXTRA_CUDA_ARCHS (e.g. "120"). With the variable unset
(gpu-a240) the build flags are exactly the original ones. Source code of the kernels is not touched.

    python tools/patch_mamba_build.py <setup.py> [...]
"""
import sys

OLD = ('    if bare_metal_version >= Version("11.8"):\n'
       '        cc_flag.append("-gencode")\n'
       '        cc_flag.append("arch=compute_90,code=sm_90")\n')
ADD = ('    for _arch in os.environ.get("PBTS_EXTRA_CUDA_ARCHS", "").split():  # PBTS patch (tools/patch_mamba_build.py)\n'
       '        cc_flag += ["-gencode", f"arch=compute_{_arch},code=sm_{_arch}"]\n')

for p in sys.argv[1:]:
    s = open(p).read()
    if ADD in s:
        print("already patched:", p)
    elif s.count(OLD) == 1 and "import os" in s:
        open(p, "w").write(s.replace(OLD, OLD + ADD))
        print("patched:", p)
    else:
        sys.exit(f"unexpected content, not patched: {p}")
