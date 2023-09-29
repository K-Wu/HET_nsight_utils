if __name__ == "__main__":
    from .. import load_raw_gpu_metric_util_report
    import os
    from .utils import is_pwd_correct_for_testing, get_path_to_test_dir

    assert is_pwd_correct_for_testing(), (
        "Please run this script at the directory where the nsight_utils is in."
        " The command will be something like python -m"
        " nsight_utils.test.test_load_raw_gpu_metric_util_report"
    )

    df = load_raw_gpu_metric_util_report(
        os.path.join(
            get_path_to_test_dir(),
            "graphiler.fb15k_RGAT.bg.breakdown.nsys-rep",
        )
    )
    print(df[df["SmActive"] > 0])
