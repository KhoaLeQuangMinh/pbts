"""Make mamba_ssm's language-model-head import optional (idempotent).

mamba_ssm 1.0.1's __init__ imports MambaLMHeadModel, which imports `transformers`. SegMamba only uses
mamba_ssm.Mamba, so the LM head is never used. Making that one import optional avoids installing transformers,
which would downgrade huggingface-hub in the training environment mid-study. No computation changes.

    python tools/patch_mamba_init.py <path/to/mamba_ssm/__init__.py> [...]
"""
import sys

OLD = "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel\n"
NEW = ("try:  # PBTS patch: the LM head needs `transformers`; SegMamba only uses Mamba (tools/patch_mamba_init.py)\n"
       "    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel\n"
       "except ImportError:\n"
       "    MambaLMHeadModel = None\n")

for p in sys.argv[1:]:
    s = open(p).read()
    if NEW in s:
        print("already patched:", p)
    elif s.count(OLD) == 1:
        open(p, "w").write(s.replace(OLD, NEW))
        print("patched:", p)
    else:
        sys.exit(f"unexpected content, not patched: {p}")
