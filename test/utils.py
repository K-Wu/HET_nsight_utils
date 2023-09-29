import sys
import os


def is_pwd_correct_for_testing():
    script_path: str = sys.argv[0]
    pwd: str = os.getcwd()
    repo_in_dir: str = os.path.dirname(
        os.path.dirname(os.path.dirname(script_path))
    )
    return pwd == repo_in_dir


def get_path_to_test_dir():
    assert is_pwd_correct_for_testing()
    test_dir: str = os.path.dirname(sys.argv[0])
    repo_in_dir: str = os.path.dirname(os.path.dirname(test_dir))
    # Return the relative path of script_path to repo_in_dir
    return os.path.relpath(test_dir, repo_in_dir)
