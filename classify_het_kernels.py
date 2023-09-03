from .run_once import run_once
from functools import lru_cache
import os
from typing import Tuple


@lru_cache(maxsize=None)
@run_once
def is_ctags_installed() -> bool:
    return os.system("ctags --version >/dev/null 2>/dev/null") == 0


def get_functions_from_ctags_table(ctags_table: str) -> set[str]:
    result = set()
    for line in ctags_table.split("\n"):
        if line.startswith("HET_") and line.endswith("f"):
            result.add(line.split("\t")[0])
    return result





def classify_fw_bw_kernel(func_pretty_name: str) -> str:
    if (
        "Delta" in func_pretty_name
        or "BckProp" in func_pretty_name
        or "_bck_" in func_pretty_name
        or "Backward" in func_pretty_name
    ):
        return "BckProp"
    else:
        if (
            "FwProp" in func_pretty_name
            or "_fw_" in func_pretty_name
            or "Forward" in func_pretty_name
        ):
            return "FwProp"
        else:
            print(f"Warning: assuming {func_pretty_name} is a forward kernel")
            return "FwProp"
