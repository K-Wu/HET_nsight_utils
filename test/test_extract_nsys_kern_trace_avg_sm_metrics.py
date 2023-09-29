if __name__ == "__main__":
    from .. import (
        is_pwd_generic_dev_root,
        get_last_nvtx_range,
        calc_avg_sm_metrics,
        get_kern_trace_overhead,
    )
    import os
    from .utils import is_pwd_correct_for_testing, get_path_to_test_dir

    assert is_pwd_correct_for_testing(), (
        "Please run this script at the directory where the nsight_utils is in."
        " The command will be something like python -m"
        " nsight_utils.test.test_extract_nsys_kern_trace_avg_sm_metrics"
    )

    file_path = os.path.join(
        get_path_to_test_dir(),
        "graphiler.fb15k_HGT.bg.breakdown.with.sm.traces.nsys-rep",
    )

    start, duration = get_last_nvtx_range(file_path)
