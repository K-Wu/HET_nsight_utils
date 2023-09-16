#!/usr/bin/env python3
# some code are from in third_party/sputnik/codegen/utils.py
import subprocess
import os
from functools import lru_cache
from .run_once import run_once
import datetime


@lru_cache(maxsize=None)
@run_once
def assert_git_exists() -> None:
    """Check if git is installed and available in the path."""
    try:
        subprocess.check_output(["git", "--version"])
    except Exception:  # any error means git is not installed
        raise OSError(
            "Git is not installed. Please install git and try again."
        )


@lru_cache(maxsize=None)
@run_once
def assert_gh_exists():
    """Check if gh is installed. If not, use `conda install gh --channel conda-forge` or refer to https://github.com/cli/cli install github-cli."""
    try:
        subprocess.check_output(["gh", "--version"])
    except Exception:  # any error means git is not installed
        raise OSError(
            "Github cli is not installed. Please install gh and try again."
        )


@lru_cache(maxsize=None)
def get_spreadsheet_url() -> str:
    """Get the SPREADSHEET_URL github repo variable by `gh variable list |grep SPREADSHEET_URL`."""
    assert_git_exists()
    assert_gh_exists()
    try:
        out = subprocess.check_output(["gh", "variable", "list"]).decode(
            "utf-8"
        )
    except Exception:
        raise OSError("Failed to run `gh variable list`.")
    for line in out.splitlines():
        if "SPREADSHEET_URL" in line:
            return line.split()[1]
    raise OSError("Failed to find SPREADSHEET_URL in `gh variable list`.")


def get_git_root_path() -> str:
    """Get the root path of the git repository."""
    assert_git_exists()
    return os.path.normpath(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("utf-8")
        .strip()
    )


def get_env_name_from_setup(het_root_path: str) -> str:
    # read hrt/script/setup_dev_env.sh and get the conda env name
    setup_script_path = os.path.join(
        het_root_path, "hrt", "script", "setup_dev_env.sh"
    )
    with open(setup_script_path, "r") as f:
        for line in f:
            if "conda activate" in line:
                return line.split(" ")[-1].strip()
    raise ValueError(
        "Fatal! Cannot find conda activate command in setup_dev_env.sh. Please"
        " check the file."
    )


def create_new_results_dir(prefix: str, results_dir: str) -> str:
    curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    new_dir = os.path.join(results_dir, prefix + curr_time)
    os.makedirs(new_dir)
    return new_dir
