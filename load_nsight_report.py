import os
import re
from .run_once import run_once
from functools import lru_cache
from typing import Tuple, Union
from .classify_het_kernels import classify_fw_bw_kernel
from .upload_benchmark_results import (
    create_worksheet,
    update_gspread,
    get_pretty_hostname,
    NameCanonicalizer,
    find_latest_subdirectory,
    NameCanonicalizer
)
import sys
import traceback
from typing import Callable


def simple_combine_nsys_csvs(
    raw_csvs: list[list[list[str]]],
) -> "list[list[str]]":
    """
    This function asserts headers are the same for all csvs, and keep only one header and merge the bodies of all csvs.
    """
    assert len(raw_csvs) > 0, "raw_csvs must not be empty"
    header = raw_csvs[0][0]
    for csv in raw_csvs:
        assert csv[0] == header, "Headers must be the same for all csvs"
    return [header] + [item for sublist in raw_csvs for item in sublist[1:]]


def _extract_info_from_file(
    file_path: str,
    suffix: str,
    canonicalize_to_str: Callable[[str], list[str]],
) -> "list[str]":
    file_path = os.path.basename(file_path)
    return canonicalize_to_str(file_path[: file_path.rfind(suffix)])


def extract_info_from_nsys(
    file_path: str,
    fmt: str = "model.dataset.flag_mul.flag_compact.ax_in.ax_out.ax_head",
) -> "list[str]":
    # model_name.dataset_name.mul_flag.compact_flag.nsys-rep.ax_in.ax_out.ax_head
    return _extract_info_from_file(
        file_path,
        ".nsys-rep",
        lambda x: NameCanonicalizer.to_list(x, fmt),
    )


def get_csv_rows_from_nsys_report(
    subdir_path: str,
    nsys_report_name: str,
    classify_het_kernel_func: Callable[[str], str],
    extract_info_from_nsys_filename: Callable[[str], list[str]],
) -> "list[list[str]]":
    raw_csvs: list[list[list[str]]] = []
    for filename in os.listdir(subdir_path):
        if filename.endswith(".nsys-rep"):
            print("extract Processing", filename)
            curr_csv: list[list[str]] = load_nsys_report(
                os.path.join(subdir_path, filename),
                nsys_report_name,
                classify_het_kernel_func,
            )
            info_from_filename: list[str] = extract_info_from_nsys_filename(
                filename
            )
            curr_csv = [info_from_filename + row for row in curr_csv]
            # For info from filename, Set INFO[idx] as the column names in header row
            for idx_col in range(len(info_from_filename)):
                curr_csv[0][idx_col] = f"INFO[{idx_col}]"
            raw_csvs.append(curr_csv)

    # Combine all csvs into one
    # csv_rows = [item for sublist in raw_csvs for item in sublist]
    csv_rows: list[list[str]] = simple_combine_nsys_csvs(raw_csvs)
    # print(csv_rows)
    return csv_rows


def upload_nsys_report(
    subdir_path: str,
    nsys_report_name: str,
    spreadsheet_url: str,
    classify_het_kernel_func: Callable[[str], str],
    filename_fmt: str,
):
    csv_rows: list[list[str]] = get_csv_rows_from_nsys_report(
        subdir_path,
        nsys_report_name,
        classify_het_kernel_func,
        lambda filename: extract_info_from_nsys(filename, filename_fmt),
    )

    # Create worksheet
    worksheet_title = f"[{get_pretty_hostname()}]{subdir_path.split('/')[-1]}"[
        :100
    ]
    try:
        worksheet = create_worksheet(spreadsheet_url, worksheet_title)
    except Exception as e:
        print("Failed to create worksheet:", e)
        print(traceback.format_exc())
        exit(1)

    # Upload
    try:
        update_gspread(csv_rows, worksheet)
    except Exception as e:
        print("Failed to upload ncu results:", e)
        print(traceback.format_exc())


@lru_cache(maxsize=None)
@run_once
def nsys_exists() -> bool:
    """Check if nsys is installed."""
    return os.system("nsys --version >/dev/null 2>/dev/null") == 0


@lru_cache(maxsize=None)
@run_once
def nsys_version_larger_than(version: str = "2023.3.1"):
    """Check if nsys version is larger than the specified version."""
    assert nsys_exists(), "nsys is not installed"
    nsys_version = os.popen("nsys --version").read().split()[-1].strip()
    return nsys_version >= version


@lru_cache(maxsize=None)
def get_nsys_recipe_package_path() -> str:
    """Get the path of the nsys recipe package."""
    assert (
        nsys_version_larger_than()
    ), "nsys older than 2023.3.1 may not support recipe"
    nsys_path = find_latest_subdirectory("/opt/nvidia/nsight-systems/", "")
    package_path = os.path.join(
        nsys_path, "target-linux-x64", "python", "packages"
    )
    return package_path


@lru_cache(maxsize=None)
def get_nsysstats_package_path() -> str:
    """Get the path of the nsysstats package."""
    assert nsys_exists(), "nsys is not installed"
    nsys_path = find_latest_subdirectory("/opt/nvidia/nsight-systems/", "")
    package_path = os.path.join(nsys_path, "target-linux-x64", "python", "lib")
    return package_path


@lru_cache(maxsize=None)
@run_once
def ncu_exists() -> bool:
    """Check if ncu is installed."""
    return os.system("ncu --version >/dev/null 2>/dev/null") == 0


def _extract_csv_from_nsys_cli_output(
    nsys_cli_output: str, classify_het_kernel_func: Callable[[str], str]
) -> "list[list[str]]":
    """Extract csv from nsys cli output."""
    import csv

    raw_csv_lines: list[str] = []
    for line in nsys_cli_output.split("\n"):
        line = line.strip()
        if len(line) == 0:
            continue
        elif re.match(
            r"Processing \[([\.\w\/\-])+\] with \[([\.\w\/\-])+\]\.\.\.", line
        ):
            continue
        elif line.find("Generating SQLite file") != -1:
            continue
        else:
            # print(line)
            # result.append(line.split(","))
            raw_csv_lines.append(line)

    # Use csv instead of .split(",") to handle the case where substring is put in double quotes, e.g., kernel name with comma
    # From https://stackoverflow.com/questions/49117525/how-to-parse-csv-with-quoted-strings-advanced-case
    raw_csv_lines = [line.replace('"', '"""') for line in raw_csv_lines]
    cr = csv.reader(raw_csv_lines, skipinitialspace=True)
    csv_rows: "list[list[str]]" = [*cr]

    # Add pretty name and HET_ID
    header = csv_rows[0]
    kernel_name_col_idx = header.index("Kernel Name")
    header.append("Pretty Name")
    header.append("HET_ID")
    het_id = 0
    for row in csv_rows[1:]:
        kernel_name = row[kernel_name_col_idx]
        pretty_name = prettify_name_from_func_signature(kernel_name)
        row.append(pretty_name)
        if classify_het_kernel_func(pretty_name) != "Non-HET Others":
            row.append(str(het_id))
            het_id += 1
        else:
            row.append("")

    return csv_rows


def extract_sqlite_from_nsys_report(filename: str) -> None:
    """Extract sqlite from nsys report file."""
    assert nsys_exists(), "nsys is not installed"
    assert os.path.exists(filename), f"{filename} does not exist"
    assert filename.endswith(".nsys-rep"), f"{filename} is not a nsys report"
    output_filename = filename[: filename.rfind(".nsys-rep")] + ".sqlite"
    if os.path.exists(output_filename):
        print(f"{output_filename} already exists")
        return
    # Use read() to make the call blocking
    os.popen(f"nsys export -t sqlite -o {output_filename} {filename}").read()
    return


def load_nsys_report(
    filename: str,
    report_name: str,
    classify_het_kernel_func: Callable[[str], str],
) -> "list[list[str]]":
    """Load a report from a nsys report file. The output, list[list[str]] is each cell in each row in the csv format output."""
    assert nsys_exists(), "nsys is not installed"
    assert os.path.exists(filename), f"{filename} does not exist"
    nsys_cli_output: str = os.popen(
        f"nsys stats -f csv -r {report_name} {filename}"
    ).read()
    return _extract_csv_from_nsys_cli_output(
        nsys_cli_output, classify_het_kernel_func
    )


NCU_DETAILS_COLUMN_IDX: "dict[str, int]" = {
    "ID": 0,
    "Kernel Name": 4,
    "Section Name": 11,
    "Metric Name": 12,
    "Metric Unit": 13,
    "Metric Value": 14,
    "Rule Name": 15,
    "Rule Type": 16,
    "Rule Description": 17,
}


def get_raw_column_idx_and_convertion(
    header: "list[str]",
    units: "list[str]",
    columns: "set[str]",
    metric_unit_conversion: "dict[Tuple[str, str], Tuple[str, int]]",
) -> Tuple["dict[str, int]", "dict[str, int]", "list[str]"]:
    raw_column_idx: dict[str, int] = {}
    exponential_to_apply: dict[str, int] = {}
    new_units: list[str] = [*units]
    for column in columns:
        raw_column_idx[column] = header.index(column)
        assert column not in header[raw_column_idx[column] + 1 :], (
            "no raw metric should be associated with two different units in"
            " the same profiling file"
        )
        if (column, units[raw_column_idx[column]]) in metric_unit_conversion:
            (
                new_unit,
                exponential_to_apply[column],
            ) = metric_unit_conversion[(column, units[raw_column_idx[column]])]
            new_units[raw_column_idx[column]] = new_unit
        else:
            exponential_to_apply[column] = 0
    return raw_column_idx, exponential_to_apply, new_units


def extract_ncu_values_from_raws(
    ncu_details_csv: "list[list[str]]",
    # It seems arithmetic intensity is AchievedWorkPerSecond/BytesPerSecond
    # It is safe to duplicate entries because raw_metrics is a set
    raw_metrics: "set[str]" = {
        # Achieved work
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",  # value per cycle (1/3)
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",  # value per cycle (2/3)
        "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",  # Predicated-On FFMA Operations Per Cycle value per cycle (3/3)
        "smsp__cycles_elapsed.avg.per_second",  # "SM Frequency" cycle per second
        # L2 achieved traffic
        "l1tex__m_xbar2l1tex_read_bytes.sum.per_second",  # L2 Cache bandwidth achieved value
        # L1 achieved traffic
        "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.per_second",  # L1 Cache Bandwidth (Global/Local) achieved traffic
        # DRAM achieved traffic
        "dram__bytes.sum.per_second",  # DRAM Bandwidth achieved value
        # Compute roofline
        "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",  # Theoretical Predicated-On FFMA Operations value per cycle
        "sm__cycles_elapsed.avg.per_second",  # "SM Frequency" cycle per second
        # DRAM roofline
        "dram__bytes.sum.peak_sustained",  # "Theoretical DRAM Bytes Accessible"
        "dram__cycles_elapsed.avg.per_second",  # DRAM frequency cycle per second
        # L1 roofline
        "derived__l1tex__lsu_writeback_bytes_mem_lg.sum.peak_sustained",  # Theoretical L1/TEX Cache Bytes Accessible
        "l1tex__cycles_elapsed.avg.per_second",  # L1 cache frequency cycle per second
        # L2 roofline
        "l1tex__m_xbar2l1tex_read_bytes.sum.peak_sustained",  # "Theoretical L2 Cache Bytes Accessible" value per cycle
        "lts__cycles_elapsed.avg.per_second",  # "L2 cache frequency" cycle per second
    },
    metric_unit_conversion: "dict[Tuple[str, str], Tuple[str, int]]" = {
        ("dram__bytes.sum.peak_sustained", "Kbyte/cycle"): ("byte/cycle", 3),
        ("dram__bytes.sum.per_second", "Tbyte/second"): ("Gbyte/second", 3),
    },
) -> "list[list[str]]":
    # TODO: use to kernel_instances_metrics and to ncu_raw_csv
    header: list[str] = ncu_details_csv[0]
    units: list[str] = ncu_details_csv[1]
    assert header[0] == "ID", f"header[0] = {header[0]} != ID"
    assert (
        header[4] == "Kernel Name"
    ), f"header[4] = {header[4]} != Kernel Name"
    NCU_DETAILS_COLUMN_IDX: dict[str, int]
    exponential_to_apply: dict[str, int]
    new_units: list[str]
    (
        NCU_DETAILS_COLUMN_IDX,
        exponential_to_apply,
        new_units,
    ) = get_raw_column_idx_and_convertion(
        header,
        units,
        raw_metrics,
        metric_unit_conversion,
    )
    # for idx in NCU_DETAILS_COLUMN_IDX.values():
    #    print(f"header[{idx}] = {header[idx]}, units[{idx}] = {units[idx]}")
    results: list[list[str]] = [
        ["ID", "Pretty Name", "Kernel Name"]
        + [key for key in NCU_DETAILS_COLUMN_IDX],
        ["", "", ""]
        + [
            new_units[NCU_DETAILS_COLUMN_IDX[key]]
            for key in NCU_DETAILS_COLUMN_IDX
        ],
    ]
    for row in ncu_details_csv[2:]:  # Skip header and units
        results.append(
            [
                row[0],  # ID
                prettify_name_from_func_signature(row[4]),  # Kernel Name
                row[4],  # Kernel Name
            ]
            + [
                str(
                    float(row[NCU_DETAILS_COLUMN_IDX[key]])
                    * 10 ** exponential_to_apply[key]
                )
                for key in NCU_DETAILS_COLUMN_IDX
            ]
        )
    return results


def reorder_columns_in_raw_csv(
    kernel_instances_per_row: "list[list[str]]",
    metric_front: "list[Tuple[str,str]]",
    metric_end: "list[Tuple[str,str]]",
) -> "list[list[str]]":
    """
    Reorder the columns in raw csv so that the first few columns are those specified in front, and the last few columns are those specified in end
    """
    header: list[str] = kernel_instances_per_row[0]
    units: list[str] = kernel_instances_per_row[1]
    header_and_units: list[Tuple[str, str]] = [
        (header[i], units[i]) for i in range(len(header))
    ]
    kernel_identifier_columns: list[Tuple[str, str]] = [
        ("ID", ""),
        ("Pretty Name", ""),
        ("Kernel Name", ""),
    ]
    new_header_and_units: list[Tuple[str, str]] = (
        kernel_identifier_columns
        + metric_front
        + list(
            set(header_and_units)
            .difference(set(metric_front))
            .difference(set(metric_end))
            .difference(set(kernel_identifier_columns))
        )
        + metric_end
    )
    column_idx_to_original_idx: list[int] = [
        header_and_units.index(ele) for ele in new_header_and_units
    ]
    results: list[list[str]] = [
        [ele[0] for ele in new_header_and_units],
        [ele[1] for ele in new_header_and_units],
    ]
    for row in kernel_instances_per_row[2:]:
        results.append([row[idx] for idx in column_idx_to_original_idx])
    return results


def get_float_metric_or_zero(
    metrics: "dict[Tuple[str, str], str]", key: Tuple[str, str]
) -> float:
    if key not in metrics:
        return 0.0
    else:
        return float(metrics[key])


def derive_rooflines(
    kernel_instances_metrics: "dict[Tuple[str, str, str], dict[Tuple[str, str], str]]",
    metrics_and_units: "set[Tuple[str, str]]",
) -> None:
    """compute rooflines and achieved values and add them to kernel_instances_metrics, and headers to metrics_and_units
    The units are from profiling results from RTX 3090. It may be subject to models and hardware.
    """
    metrics_and_units.add(("Achieved Work", "GFLOPs"))
    metrics_and_units.add(("Compute Roofline", "GFLOPs"))
    metrics_and_units.add(("DRAM Roofline", "Gbyte/second"))
    metrics_and_units.add(("DRAM Achieved Traffic", "Gbyte/second"))

    for kernel_identifier in kernel_instances_metrics:
        # Achieved work
        # terms of flops_per_cycle are always inst/cycle
        peak_flop_per_cycle: float = get_float_metric_or_zero(
            kernel_instances_metrics[kernel_identifier],
            (
                "derived__sm__sass_thread_inst_executed_op_ffma_pred_on_x2",
                "inst",
            ),
        )
        flop_per_cycle: float = (
            get_float_metric_or_zero(
                kernel_instances_metrics[kernel_identifier],
                (
                    "derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2",
                    "inst",  # This is what ncu reports, instead of inst/cycle
                ),
            )
            + get_float_metric_or_zero(
                kernel_instances_metrics[kernel_identifier],
                (
                    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed",
                    "inst/cycle",
                ),
            )
            + get_float_metric_or_zero(
                kernel_instances_metrics[kernel_identifier],
                (
                    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed",
                    "inst/cycle",
                ),
            )
        )  # warp size 32
        sm_cycle_per_nano_second: float = get_float_metric_or_zero(
            kernel_instances_metrics[kernel_identifier],
            ("smsp__cycles_elapsed.avg.per_second", "cycle/nsecond"),
        )
        kernel_instances_metrics[kernel_identifier][
            ("Achieved Work", "GFLOPs")
        ] = str(flop_per_cycle * sm_cycle_per_nano_second)
        # print(str(
        #     flop_per_cycle * sm_cycle_per_nano_second
        # ))
        kernel_instances_metrics[kernel_identifier][
            ("Compute Roofline", "GFLOPs")
        ] = str(
            peak_flop_per_cycle * sm_cycle_per_nano_second
        )  # warp size * 32

        dram_peak_bandwidth: float = get_float_metric_or_zero(
            kernel_instances_metrics[kernel_identifier],
            ("dram__bytes.sum.peak_sustained", "byte/cycle"),
        )
        if dram_peak_bandwidth == 0.0:  # A100 special handling
            dram_peak_bandwidth = (
                get_float_metric_or_zero(
                    kernel_instances_metrics[kernel_identifier],
                    ("dram__bytes.sum.peak_sustained", "Kbyte/cycle"),
                )
                * 1000
            )
        dram_cycle_per_second: float = get_float_metric_or_zero(
            kernel_instances_metrics[kernel_identifier],
            ("dram__cycles_elapsed.avg.per_second", "cycle/nsecond"),
        )
        dram_achieved_traffic = get_float_metric_or_zero(
            kernel_instances_metrics[kernel_identifier],
            ("dram__bytes.sum.per_second", "Gbyte/second"),
        )
        if dram_achieved_traffic == 0.0:  # A100 special handling
            dram_achieved_traffic = (
                get_float_metric_or_zero(
                    kernel_instances_metrics[kernel_identifier],
                    ("dram__bytes.sum.per_second", "Tbyte/second"),
                )
                * 1000
            )
        kernel_instances_metrics[kernel_identifier][
            ("DRAM Roofline", "Gbyte/second")
        ] = str(dram_peak_bandwidth * dram_cycle_per_second)
        kernel_instances_metrics[kernel_identifier][
            ("DRAM Achieved Traffic", "Gbyte/second")
        ] = str(dram_achieved_traffic)


UNITS_TO_EXPONENTIAL: dict[str, int] = {
    "G": 9,
    "M": 6,
    "T": 12,
    "K": 3,
    "u": -6,
    "n": -9,
}

EXPONENTIAL_TO_UNITS: dict[int, str] = {
    UNITS_TO_EXPONENTIAL[key]: key for key in UNITS_TO_EXPONENTIAL
}
EXPONENTIAL_TO_UNITS[0] = ""


def derive_kernel_categories(
    kernel_instances_metrics: "dict[Tuple[str, str, str], dict[Tuple[str, str], str]]",
    metrics_and_units: "set[Tuple[str, str]]",
    classify_het_kernel_func: Callable[[str], str],
) -> None:
    metrics_and_units.add(("Kernel Category", ""))
    for kernel_identifier in kernel_instances_metrics:
        kernel_instances_metrics[kernel_identifier][
            ("Kernel Category", "")
        ] = classify_het_kernel_func(
            kernel_identifier[1]
        )  # kernel_identifier[2] is pretty name


def derive_kernel_forward_or_backward(
    kernel_instances_metrics: "dict[Tuple[str, str, str], dict[Tuple[str, str], str]]",
    metrics_and_units: "set[Tuple[str, str]]",
) -> None:
    metrics_and_units.add(("Kernel Forward or Backward", ""))
    for kernel_identifier in kernel_instances_metrics:
        kernel_instances_metrics[kernel_identifier][
            ("Kernel Forward or Backward", "")
        ] = classify_fw_bw_kernel(
            kernel_identifier[2]
        )  # kernel_identifier[2] is pretty name


def consolidate_ncu_details(
    metric_per_row: "list[list[str]]",
    classify_het_kernel_func: Union[Callable[[str], str], None],
) -> "list[list[str]]":
    """
    The original output from extract_ncu_values_from_details shows one metric in each row,
    this function consolidate it so that each row show all metrics of a kernel instance,
    similar to the ncu raw csv output
    """
    header: list[str] = metric_per_row[0]
    name_columns: list[str] = ["ID", "Pretty Name", "Kernel Name"]
    name_columns_idx: dict[str, int] = {
        key: header.index(key) for key in name_columns
    }
    metric_columns: list[str] = ["Metric Name", "Metric Unit", "Metric Value"]
    metric_columns_idx: dict[str, int] = {
        key: header.index(key) for key in metric_columns
    }
    kernel_instances_metrics: dict[
        Tuple[str, str, str], dict[Tuple[str, str], str]
    ] = {}

    metrics_and_units: set[Tuple[str, str]] = set()
    for row in metric_per_row[1:]:
        kernel_identifier: tuple[str, str, str] = (
            row[name_columns_idx["ID"]],
            row[name_columns_idx["Pretty Name"]],
            row[name_columns_idx["Kernel Name"]],
        )
        if kernel_identifier not in kernel_instances_metrics:
            kernel_instances_metrics[kernel_identifier] = dict()
        assert (
            row[metric_columns_idx["Metric Name"]],
            row[metric_columns_idx["Metric Unit"]],
        ) not in kernel_instances_metrics[
            kernel_identifier
        ], f"Duplicate metric: {row}"

        kernel_instances_metrics[kernel_identifier][
            (
                row[metric_columns_idx["Metric Name"]],
                row[metric_columns_idx["Metric Unit"]],
            )
        ] = row[metric_columns_idx["Metric Value"]]

        metrics_and_units.add(
            (
                row[metric_columns_idx["Metric Name"]],
                row[metric_columns_idx["Metric Unit"]],
            )
        )

    if classify_het_kernel_func is not None:
        derive_kernel_categories(
            kernel_instances_metrics,
            metrics_and_units,
            classify_het_kernel_func,
        )
    derive_kernel_forward_or_backward(
        kernel_instances_metrics, metrics_and_units
    )

    results: list[list[str]] = [
        name_columns
        + [ele[0] for ele in sorted(metrics_and_units, reverse=True)],
        [""] * len(name_columns)
        + [ele[1] for ele in sorted(metrics_and_units, reverse=True)],
    ]
    for kernel_identifier in kernel_instances_metrics:
        row = list(kernel_identifier)
        for metric, unit in sorted(metrics_and_units, reverse=True):
            if (metric, unit) not in kernel_instances_metrics[
                kernel_identifier
            ]:
                row.append("")
            else:
                row.append(
                    kernel_instances_metrics[kernel_identifier][(metric, unit)]
                )
        results.append(row)
    assert "Metric Name" not in results[0]
    return results


def convert_kernel_instances_metrics_to_ncu_raw_csv(
    kernel_instances_metrics: "dict[Tuple[str, str, str], dict[Tuple[str, str], str]]",
    metrics_and_units: "set[Tuple[str, str]]",
) -> "list[list[str]]":
    result_header: list[str] = ["ID", "Pretty Name", "Kernel Name"] + [
        ele[0] for ele in sorted(metrics_and_units, reverse=True)
    ]
    result_units: list[str] = [""] * 3 + [
        ele[1] for ele in sorted(metrics_and_units, reverse=True)
    ]
    results: list[list[str]] = [result_header, result_units]
    for kernel_identifier in kernel_instances_metrics:
        row = list(kernel_identifier)
        for metric, unit in sorted(metrics_and_units, reverse=True):
            if (metric, unit) not in kernel_instances_metrics[
                kernel_identifier
            ]:
                row.append("")
            else:
                row.append(
                    kernel_instances_metrics[kernel_identifier][(metric, unit)]
                )
        results.append(row)
    return results


def convert_ncu_raw_csvs_to_kernel_instances_metrics(
    raw_csv: "list[list[str]]",
) -> Tuple[
    "set[Tuple[str, str]]",
    "dict[Tuple[str, str, str], dict[Tuple[str, str], str]]",
]:
    kernel_instances_metrics: dict[
        Tuple[str, str, str], dict[Tuple[str, str], str]
    ] = {}
    metrics_and_units: set[Tuple[str, str]] = set()
    header: list[str] = raw_csv[0]
    units: list[str] = raw_csv[1]
    assert header[0] == "ID", f"header[0] = {header[0]} != ID"
    assert (
        header[1] == "Pretty Name"
    ), f"header[1] = {header[1]} != Pretty Name"
    assert (
        header[2] == "Kernel Name"
    ), f"header[2] = {header[2]} != Kernel Name"
    for row in raw_csv[2:]:
        kernel_identifier: tuple[str, str, str] = (
            row[0],
            row[1],
            row[2],
        )
        if kernel_identifier not in kernel_instances_metrics:
            kernel_instances_metrics[kernel_identifier] = dict()
        for metric_idx in range(3, len(row)):
            curr_metric = header[metric_idx]
            curr_unit = units[metric_idx]
            curr_value = row[metric_idx]
            if (curr_metric, curr_unit) not in kernel_instances_metrics[
                kernel_identifier
            ]:
                kernel_instances_metrics[kernel_identifier][
                    (curr_metric, curr_unit)
                ] = curr_value
                metrics_and_units.add((curr_metric, curr_unit))
            else:
                print(
                    "Warning: duplicate metric",
                    curr_metric,
                    curr_unit,
                    curr_value,
                    kernel_identifier,
                    kernel_instances_metrics[kernel_identifier][
                        (curr_metric, curr_unit)
                    ],
                )
    return metrics_and_units, kernel_instances_metrics


def calculate_roofline_for_ncu_raw_csvs(
    raw_csv: "list[list[str]]",
) -> "list[list[str]]":
    (
        metrics_and_units,
        kernel_instances_metrics,
    ) = convert_ncu_raw_csvs_to_kernel_instances_metrics(raw_csv)
    derive_rooflines(kernel_instances_metrics, metrics_and_units)
    return convert_kernel_instances_metrics_to_ncu_raw_csv(
        kernel_instances_metrics, metrics_and_units
    )


def combine_ncu_raw_csvs(
    num_frozen_columns: int,
    raw_csv_list: "list[list[list[str]]]",
) -> "list[list[str]]":
    """
    Combine multiple raw csvs from ncu into one
    the first few columns won't be touched, i.e., model, dataset, ID, pretty name, kernel name
    the number of frozen columns is specified by num_frozen_columns
    Headers will be merged into one.
    """
    assert len(raw_csv_list) > 0
    kernel_instances_metrics: dict[
        Tuple[str, ...], dict[Tuple[str, str], str]
    ] = {}
    metrics_and_units: set[Tuple[str, str]] = set()
    for raw_csv_ in raw_csv_list:
        header: list[str] = raw_csv_[0]
        units: list[str] = raw_csv_[1]
        assert (
            header[num_frozen_columns - 3] == "ID"
        ), f"header[0] = {header[0]} != ID"
        assert (
            header[num_frozen_columns - 2] == "Pretty Name"
        ), f"header[1] = {header[1]} != Pretty Name"
        assert (
            header[num_frozen_columns - 1] == "Kernel Name"
        ), f"header[2] = {header[2]} != Kernel Name"
        for row in raw_csv_[2:]:
            kernel_identifier: tuple[str, ...] = tuple(
                row[:num_frozen_columns]
            )
            if kernel_identifier not in kernel_instances_metrics:
                kernel_instances_metrics[kernel_identifier] = dict()
            # Metric columns start from num_frozen_columns
            for metric_idx in range(num_frozen_columns, len(row)):
                curr_metric = header[metric_idx]
                curr_unit = units[metric_idx]
                curr_value = row[metric_idx]
                if (curr_metric, curr_unit) not in kernel_instances_metrics[
                    kernel_identifier
                ]:
                    kernel_instances_metrics[kernel_identifier][
                        (curr_metric, curr_unit)
                    ] = curr_value
                    metrics_and_units.add((curr_metric, curr_unit))
                else:
                    print(
                        "Warning: duplicate metric",
                        curr_metric,
                        curr_unit,
                        curr_value,
                        kernel_identifier,
                        kernel_instances_metrics[kernel_identifier][
                            (curr_metric, curr_unit)
                        ],
                    )

    # using stale values
    result_header: list[str] = header[:num_frozen_columns] + [
        ele[0] for ele in sorted(metrics_and_units, reverse=True)
    ]
    result_units: list[str] = units[:num_frozen_columns] + [
        ele[1] for ele in sorted(metrics_and_units, reverse=True)
    ]
    results: list[list[str]] = [result_header, result_units]
    for kernel_identifier in kernel_instances_metrics:
        row = list(kernel_identifier)
        for metric, unit in sorted(metrics_and_units, reverse=True):
            if (metric, unit) not in kernel_instances_metrics[
                kernel_identifier
            ]:
                row.append("")
            else:
                row.append(
                    kernel_instances_metrics[kernel_identifier][(metric, unit)]
                )
        results.append(row)
    return results


def unit_to_str(
    exponential: int, nominator: "list[str]", denominator: "list[str]"
) -> str:
    assert len(nominator) <= 1, f"nominator = {nominator} is not a single unit"
    assert (
        len(denominator) <= 1
    ), f"denominator = {denominator} is not a single unit"
    nominator_str = "" if len(nominator) == 0 else nominator[0]
    if len(denominator) == 0:
        return EXPONENTIAL_TO_UNITS[exponential] + nominator_str
    else:
        return (
            EXPONENTIAL_TO_UNITS[exponential]
            + nominator_str
            + "/"
            + denominator[0]
        )


def mul_two_units(lhs: str, rhs: str) -> str:
    # mul_two_units("cycle/nsecond", "Kbyte/cycle") = "Tbyte/second"
    return unit_to_str(
        *_mul_two_units(canonicalize_unit(lhs), canonicalize_unit(rhs))
    )


def canonicalize_unit(unit: str) -> Tuple[int, "list[str]", "list[str]"]:
    # extract exponential, numerator, denominator from unit
    if len(unit.split("/")) > 1:
        numerator = canonicalize_unit(unit.split("/")[0])
        denominator = canonicalize_unit(unit.split("/")[1])
        return _div_two_units(numerator, denominator)
    exponential = 0
    if unit[0] in UNITS_TO_EXPONENTIAL:
        exponential += UNITS_TO_EXPONENTIAL[unit[0]]
    return exponential, [unit[1:]], []


def _simplify_unit_fraction(
    nominator: "list[str]", denominator: "list[str]"
) -> Tuple["list[str]", "list[str]"]:
    # simplify the fraction
    for idx in range(len(nominator)):
        if nominator[idx] in denominator:
            nominator[idx] = ""
            denominator[denominator.index(nominator[idx])] = ""
    nominator = [ele for ele in nominator if len(ele) > 0]
    denominator = [ele for ele in denominator if len(ele) > 0]
    return nominator, denominator


def _div_two_units(
    lhs: Tuple[int, "list[str]", "list[str]"],
    rhs: Tuple[int, "list[str]", "list[str]"],
) -> Tuple[int, "list[str]", "list[str]"]:
    nominator = lhs[1] + rhs[2]
    denominator = rhs[1] + lhs[2]
    nominator, denominator = _simplify_unit_fraction(nominator, denominator)
    return lhs[0] - rhs[0], nominator, denominator


def _mul_two_units(
    lhs: Tuple[int, "list[str]", "list[str]"],
    rhs: Tuple[int, "list[str]", "list[str]"],
) -> Tuple[int, "list[str]", "list[str]"]:
    nominator = lhs[1] + rhs[1]
    denominator = rhs[2] + lhs[2]
    nominator, denominator = _simplify_unit_fraction(nominator, denominator)
    return lhs[0] + rhs[0], nominator, denominator


def div_two_units(lhs: str, rhs: str) -> str:
    # div_two_units("Tbyte", "cycle/nsecond") = "Kbyte/cycle"
    return unit_to_str(
        *_div_two_units(canonicalize_unit(lhs), canonicalize_unit(rhs))
    )


def extract_ncu_values_from_details(
    ncu_details_csv: "list[list[str]]",
    metric_names: "set[str]" = {
        "ID",
        "L2 Cache Throughput",
        "L1/TEX Cache Throughput",
        "DRAM Throughput",  # dram__bytes.sum.per_second
        "Memory Throughput",  # unit: "%", "Gbyte/second", "Kbyte/second"
        "Elapsed Cycles",
        "Issued Warp Per Scheduler",  # unit: ""
        "Compute (SM) Throughput",  # unit: "%"
        "Achieved Occupancy",  # unit: "%"
        "Achieved Active Warps Per SM",  # unit: "warp"
        "Executed Ipc Active",  # unit: "inst/cycle"
        "Executed Ipc Elapsed",  # unit: "inst/cycle"
        "Duration",
    },
    metric_unit_conversion: "dict[Tuple[str, str], Tuple[str, int]]" = {
        ("Memory Throughput", "Kbyte/second"): ("Gbyte/second", -6),
        ("Memory Throughput", "Mbyte/second"): ("Gbyte/second", -3),
        ("Memory Throughput", "Tbyte/second"): ("Gbyte/second", 3),
        ("Duration", "msecond"): ("usecond", 3),
        ("Duration", "nsecond"): ("usecond", -3),
    },
) -> "list[list[str]]":
    header: list[str] = ncu_details_csv[0]

    results: list[list[str]] = [
        [
            "ID",
            "Pretty Name",
            "Kernel Name",
            "Metric Name",
            "Metric Unit",
            "Metric Value",
        ]
    ]
    for key in NCU_DETAILS_COLUMN_IDX:
        assert header[NCU_DETAILS_COLUMN_IDX[key]] == key, (
            f"header[{NCU_DETAILS_COLUMN_IDX[key]}] ="
            f" {header[NCU_DETAILS_COLUMN_IDX[key]]} != {key}"
        )
    for row in ncu_details_csv[1:]:
        if row[NCU_DETAILS_COLUMN_IDX["Metric Name"]] in metric_names:
            curr_metric_name = row[NCU_DETAILS_COLUMN_IDX["Metric Name"]]
            curr_metric_unit = row[NCU_DETAILS_COLUMN_IDX["Metric Unit"]]
            curr_metric_value = row[NCU_DETAILS_COLUMN_IDX["Metric Value"]]
            if (curr_metric_name, curr_metric_unit) in metric_unit_conversion:
                (
                    curr_metric_unit,
                    exponential_to_apply,
                ) = metric_unit_conversion[
                    (curr_metric_name, curr_metric_unit)
                ]
                curr_metric_value = str(
                    float(curr_metric_value) * 10**exponential_to_apply
                )
            results.append(
                [
                    row[NCU_DETAILS_COLUMN_IDX["ID"]],
                    prettify_name_from_func_signature(
                        row[NCU_DETAILS_COLUMN_IDX["Kernel Name"]]
                    ),
                    row[NCU_DETAILS_COLUMN_IDX["Kernel Name"]],
                    row[NCU_DETAILS_COLUMN_IDX["Metric Name"]],
                    curr_metric_unit,
                    curr_metric_value,
                ]
            )
    return results


def load_csv_from_multiline_string(csv_string: str) -> "list[list[str]]":
    # ncu output is multiline csv where each cell value is wrapped by double quotes.
    # We need to remove the double quotes and split the string by comma.
    result: list[list[str]] = []
    lines: list[str] = csv_string.split("\n")
    for line in lines:
        line: str = line.strip()
        if len(line) == 0:
            continue
        elif line.startswith('"'):
            if line.endswith('"'):
                result.append(line[1:-1].split('","'))
            elif line.endswith('",'):
                result.append(line[1:-2].split('","'))
            continue

        print(
            'Warning:  line does not start with " or end with " skipping:',
            line,
        )
    return result


def load_ncu_report(filename: str, page_name: str) -> "list[list[str]]":
    """Load a report from a ncu report file."""
    assert ncu_exists(), "ncu is not installed"
    assert os.path.exists(filename), f"{filename} does not exist"
    ncu_cli_output: str = os.popen(
        f"ncu --page {page_name} --csv  --import {filename}"
    ).read()
    return load_csv_from_multiline_string(ncu_cli_output)


# TODO: handle the cases where <unnamed>:: or ::<unnamed>:: causes name substring after :: to be truncated
def prettify_name_from_func_signature(func_signature: str) -> str:
    # func_signature: HET_XXX<XX,XX,XX>(XXX, XXX, XXX)
    result: str = (
        func_signature.split("(")[0]
        .strip()
        .split("<")[0]
        .strip()
        .split(" ")[-1]
        .strip()
    )
    if result == "cutlass::Kernel":
        return (
            func_signature.split("(")[0].strip().split("<")[1].strip()[:-1]
        )  # remove the last >
    else:
        return result


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


def check_metric_units_all_identical_from_ncu_folder(path: str) -> bool:
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
