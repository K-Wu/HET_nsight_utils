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
    if os.path.exists(new_dir):
        # Try again with seconds
        curr_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_dir = os.path.join(results_dir, prefix + curr_time)

    # Purposefully throw error if the directory already exists
    os.makedirs(new_dir)
    return new_dir


def is_generic_root_path(path: str, repo_title: str) -> bool:
    try:
        # Check if we can find title as HET in README.md
        # File not found error during cat cannot be catched. So we use os.path.exists to check first
        if not os.path.exists(os.path.join(path, "README.md")):
            return False
        res = subprocess.check_output(["cat", os.path.join(path, "README.md")])
        res = res.decode("utf-8")
        if "# " + repo_title in res:
            return True
        elif "## What's in a name?" in res:
            raise ValueError(
                "Fatal! Detected sub-header, What's in a name, in README.md"
                f" but not found # {repo_title}. Is the top-level project"
                " renamed? Please update it in the detect_pwd.py, or avoid"
                " using the subheading in a non-top-level project."
            )
        return False
    except OSError:
        return False


def is_pwd_generic_dev_root(repo_title: str, devpath_basename: str) -> bool:
    """Return if pwd is get_generic_root_path({repo_title})/{devpath_basename}"""
    return (
        is_generic_root_path(os.path.dirname(os.getcwd()), repo_title)
        and os.path.basename(os.getcwd()) == devpath_basename
    )


def get_generic_root_path(repo_title: str) -> str:
    """Go to the root path of the git repository, and go to the parent directory until we find the root path with the specified repo_title"""
    path = get_git_root_path()
    while not is_generic_root_path(path, repo_title):
        path = os.path.dirname(path)
    return path
