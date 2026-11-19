"""Locked-test guard. Any code path that reads a case calls check_not_locked() unless it is the
single, deliberate final evaluation (allow_locked=True, which also writes an audit line)."""
import datetime
import getpass
import json
from functools import lru_cache

from pbts import config as C


@lru_cache(maxsize=1)
def locked_cases() -> frozenset:
    return frozenset(json.load(open(C.LOCKED_DIR / "cases.json"))["locked"])


def check_not_locked(cases, allow_locked: bool = False, purpose: str = ""):
    hit = sorted(set(cases) & locked_cases())
    if hit and not allow_locked:
        raise PermissionError(f"{len(hit)} locked-test case(s) requested (e.g. {hit[:3]}). "
                              "The locked test may only be used in the final evaluation.")
    if hit and allow_locked:
        with open(C.LOCKED_DIR / "ACCESS_LOG.txt", "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {getpass.getuser()} {len(hit)} cases: {purpose}\n")
