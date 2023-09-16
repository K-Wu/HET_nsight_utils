from .run_once import run_once
from .detect_pwd import (
    get_git_root_path,
    get_env_name_from_setup,
    create_new_results_dir,
    assert_git_exists,
)
from .upload_benchmark_results import (
    ConfigCanonicalizer,
    generate_filename,
    ask_file,
    ask_subdirectory,
    ask_subdirectory_or_file,
    update_gspread,
    create_worksheet,
    get_cell_range_from_A1,
    count_cols,
    count_rows,
    get_pretty_hostname,
    find_latest_subdirectory_or_file,
    NameCanonicalizer,
)
from .classify_het_kernels import (
    is_ctags_installed,
    classify_fw_bw_kernel,
    get_functions_from_ctags_table,
)
from .load_nsight_report import (
    extract_ncu_values_from_details,
    load_ncu_report,
    upload_nsys_report,
    reorder_columns_in_raw_csv,
    extract_ncu_values_from_raws,
    calculate_roofline_for_ncu_raw_csvs,
    load_nsys_report,
    consolidate_ncu_details,
    prettify_name_from_func_signature,
)
from .upload_cache_and_roofline import (
    extract_from_ncu_folder,
    extract_from_ncu_file,
)
from .print_machine_info import (
    print_system_info,
    print_python_env_info,
    print_conda_envs_info,
)

# From https://stackoverflow.com/questions/59167405/flake8-ignore-only-f401-rule-in-entire-file
__all__ = [
    "run_once",
    "get_git_root_path",
    "get_env_name_from_setup",
    "create_new_results_dir",
    "assert_git_exists",
    "ConfigCanonicalizer",
    "generate_filename",
    "ask_file",
    "ask_subdirectory",
    "ask_subdirectory_or_file",
    "update_gspread",
    "create_worksheet",
    "get_cell_range_from_A1",
    "count_cols",
    "count_rows",
    "get_pretty_hostname",
    "find_latest_subdirectory_or_file",
    "NameCanonicalizer",
    "is_ctags_installed",
    "classify_fw_bw_kernel",
    "get_functions_from_ctags_table",
    "extract_ncu_values_from_details",
    "load_ncu_report",
    "upload_nsys_report",
    "reorder_columns_in_raw_csv",
    "extract_ncu_values_from_raws",
    "calculate_roofline_for_ncu_raw_csvs",
    "load_nsys_report",
    "consolidate_ncu_details",
    "prettify_name_from_func_signature",
    "extract_from_ncu_folder",
    "extract_from_ncu_file",
    "print_system_info",
    "print_python_env_info",
    "print_conda_envs_info",
]  # this suppresses the warning F401
