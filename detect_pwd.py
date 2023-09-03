#!/usr/bin/env python3
# some code are from in third_party/sputnik/codegen/utils.py
import subprocess
import os
from functools import lru_cache
from .run_once import run_once


@lru_cache(maxsize=None)
@run_once
def assert_git_exists() -> None:
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except Exception:  # any error means git is not installed
        raise OSError("Git is not installed. Please install git and try again.")


def get_git_root_path() -> str:
    """Get the root path of the git repository."""
    assert_git_exists()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def get_env_name_from_setup(het_root_path: str) -> str:
    # read hetero_edgesoftmax/script/setup_dev_env.sh and get the conda env name
    setup_script_path = os.path.join(
        het_root_path, "hetero_edgesoftmax", "script", "setup_dev_env.sh"
    )
    with open(setup_script_path, "r") as f:
        for line in f:
            if "conda activate" in line:
                return line.split(" ")[-1].strip()
    raise ValueError(
        "Fatal! Cannot find conda activate command in setup_dev_env.sh. Please check the file."
    )
