from .load_nsight_report import (
    extract_ncu_values_from_details,
    extract_ncu_values_from_raws,
    load_ncu_report,
    calculate_roofline_for_ncu_raw_csvs,
    combine_ncu_raw_csvs,
    consolidate_ncu_details,
    _extract_info_from_file,
)
from .upload_benchmark_results import NameCanonicalizer
import os
from typing import Callable


def extract_info_from_ncu(file_path: str) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.ncu-rep
    return _extract_info_from_file(
        file_path,
        ".ncu-rep",
        lambda x: NameCanonicalizer.to_list(
            x, "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head"
        ),
    )


def extract_from_ncu_file(
    file_path: str,
    extract_mem_flag: bool,
    extract_roofline_flag: bool,
    classify_het_kernel_func: Callable[[str], str],
) -> "list[list[str]]":
    assert file_path.endswith(".ncu-rep"), "filename must end with .ncu-rep"
    info_from_filename: list[str] = extract_info_from_ncu(file_path)
    func_and_metric_csvs: list[list[list[str]]] = []
    if extract_mem_flag:
        func_and_metric_csvs.append(
            consolidate_ncu_details(
                extract_ncu_values_from_details(
                    load_ncu_report(file_path, "details")
                ),
                classify_het_kernel_func,
            )
        )
    if extract_roofline_flag:
        func_and_metric_csvs.append(
            calculate_roofline_for_ncu_raw_csvs(
                extract_ncu_values_from_raws(load_ncu_report(file_path, "raw"))
            )
        )
    if len(func_and_metric_csvs) == 1:
        func_and_metric = func_and_metric_csvs[0]
    else:
        # Combine csvs if there are multiple csvs
        # Number of frozen columns is 3, i.e., (id, pretty name, kernel name)
        # Names and infos will be added after the combination
        func_and_metric = combine_ncu_raw_csvs(3, func_and_metric_csvs)

    # Add info_from_filename to the beginning of each row except for headers
    results = [info_from_filename + f_ for f_ in func_and_metric]
    for idx_row in range(2):
        for idx_col in range(len(info_from_filename)):
            if idx_row == 0:
                # Header name row
                results[idx_row][idx_col] = f"INFO[{idx_col}]"
            else:
                # Header unit row
                results[idx_row][idx_col] = ""
    return results


def extract_from_ncu_folder(
    path: str,
    extract_mem_flag: bool,
    extract_roofline_flag: bool,
    classify_het_kernel_func: Callable[[str], str],
) -> "list[list[str]]":
    raw_csvs: list[list[list[str]]] = []
    len_info_from_filename: int = -1
    for filename in os.listdir(path):
        print("extract_from_ncu_folder Processing", filename)
        if filename.endswith(".ncu-rep"):
            raw_csvs.append(
                extract_from_ncu_file(
                    os.path.join(path, filename),
                    extract_mem_flag,
                    extract_roofline_flag,
                    classify_het_kernel_func,
                )
            )
        if (
            len(extract_info_from_ncu(filename)) != len_info_from_filename
            and len_info_from_filename != -1
        ):
            raise ValueError("Number of frozen columns not consistent")
        len_info_from_filename = len(extract_info_from_ncu(filename))

    # number of frozen columns equals to the number of columns in info_from_filename and (id, pretty name, kernel name)
    return combine_ncu_raw_csvs(len_info_from_filename + 3, raw_csvs)
    # return [item for sublist in raw_csvs for item in sublist]


def check_metric_units_all_identical_from_ncu_folder(path) -> bool:
    """
    check_metric_units_all_identical_from_ncu_folder("misc/artifacts/ncu_breakdown_202307180518") returns False after printing
    Metric derived__memory_l1_wavefronts_shared_excessive has different units: {'Kbyte', 'byte', 'Mbyte'}
    """
    metric_units: dict[str, set[str]] = dict()
    for filename in os.listdir(path):
        if filename.endswith(".ncu-rep"):
            raw_csv: list[list[str]] = load_ncu_report(
                os.path.join(path, filename), "raw"
            )

            for idx in range(len(raw_csv[0])):
                metric: str = raw_csv[0][idx]
                unit: str = raw_csv[1][idx]
                if metric not in metric_units:
                    metric_units[metric] = set()
                metric_units[metric].add(unit)

    for metric in metric_units:
        if len(metric_units[metric]) != 1:
            if len(metric_units[metric]) == 2 and "%" in metric_units[metric]:
                continue
            print(
                f"Metric {metric} has different units: {metric_units[metric]}"
            )
            return False
    return True
