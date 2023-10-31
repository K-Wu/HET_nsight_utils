if __name__ == "__main__":
    from .. import get_worksheet_gid, open_worksheet

    import os
    from .utils import is_pwd_correct_for_testing, get_path_to_test_dir

    assert is_pwd_correct_for_testing(), (
        "Please run this script at the directory where the nsight_utils is in."
        " The command will be something like python -m"
        " nsight_utils.test.test_extract_nsys_kern_trace_avg_sm_metrics"
    )

    sheet_url = "https://docs.google.com/spreadsheets/d/1FP5IDW7hIBESdeBHztuoAdzWdPMYLm28HHrcQpfXfEk/edit?usp=sharing"
    gid = get_worksheet_gid(sheet_url, "SheetForGIDRetrievalTest")
    worksheet = open_worksheet(sheet_url, gid, False)
    print(gid)
