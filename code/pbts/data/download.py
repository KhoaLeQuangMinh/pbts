"""Step 00a: download the study data from Kaggle to <root>/data/kagglehub_cache, where config.KAGGLE_DIR expects it.

    python -m pbts.data.download

Needs your own Kaggle API token in <root>/secrets/kaggle_token (chmod 600; or KAGGLE_API_TOKEN set), and the
BraTS-PEDs data terms accepted on Kaggle. Same call as the one-off download.py used on gpu-a240 (2026-11-19) and
blackwell (2026-11-20), added to the repo on 2026-11-21. Next step: python -m pbts.data.convert
"""
import os

from pbts import config as C

KAGGLE_DATASET = "ckhangnguyn/pbts-preprocessed"


def main():
    os.environ.setdefault("KAGGLEHUB_CACHE", str(C.ROOT / "data/kagglehub_cache"))
    token = C.ROOT / "secrets/kaggle_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token.exists():
        os.environ["KAGGLE_API_TOKEN"] = token.read_text().strip()
    import kagglehub
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("DONE", path, flush=True)
    if not C.MANIFEST.exists():
        raise SystemExit(f"download finished but {C.MANIFEST} is missing: check KAGGLEHUB_CACHE and the dataset version")


if __name__ == "__main__":
    main()
