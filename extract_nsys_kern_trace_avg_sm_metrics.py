from __future__ import annotations
from typing import Callable
from .load_nsight_report import (
    prettify_name_from_func_signature,
    load_nsys_report,
)
from typing import Union
import pandas
from .nsys_metrics_report import load_raw_gpu_metric_util_report


def calc_avg_sm_metrics(
    df: "pandas.DataFrame",
    start_timestamp: int,
    end_timestamp: int,
    metric_name: str = "SmActive",
) -> float:
    """
    This function calculate the average SM metrics during the time between start_timestamp and end_timestamp.
    start_timestamp, end_timestamp: These two timestamps are the raw timestamps in nsys traces. The pair can either mark the beginning and end of a kernel, or a nvtx range.
    """
    # Use pandas.Series.searchsorted. The return index points to the first element that is greater than or equal to the searched value.
    # >>> ser = pd.Series([1, 2, 3])
    # >>> ser.searchsorted(1) # 0
    # >>> ser.searchsorted(1.5) # 1
    # >>> ser.searchsorted(2) # 1
    # >>> ser.searchsorted(2.5) # 2

    # First, find [lhs, rhs] timestamp that covers the whole kernel
    # The rhs timestamp can be obtained by rhs=ser.searchsorted(endtime)
    # For the lhs timestamp, lhs=ser.searchsorted(starttime) if ser[ser.searchsorted(starttime)]==starttime else ser.searchsorted(starttime)-1
    lhs_row_idx = (
        df["rawTimestamp"].searchsorted(start_timestamp)
        if df["rawTimestamp"][df["rawTimestamp"].searchsorted(start_timestamp)]
        == start_timestamp
        else df["rawTimestamp"].searchsorted(start_timestamp) - 1
    )
    rhs_row_idx = df["rawTimestamp"].searchsorted(end_timestamp)
    # rhs should be larger than lhs
    collection_cycle_ns = (
        df["rawTimestamp"][rhs_row_idx] - df["rawTimestamp"][lhs_row_idx] + 0.0
    ) / (rhs_row_idx - lhs_row_idx)

    # TODO: the second range should be rhs_row_idx - 1
    sm_active_ns_product: float = 0.0
    # The for loop is cut into two parts: notice that collection_start < start or end< collection_end could only happen when row_idx is in set(range(lhs_row_idx, rhs_row_idx)).difference(set(range(lhs_row_idx+1, rhs_row_idx-1))). For other cases, i.e., row_idx is in range(lhs_row_idx+1, rhs_row_idx-1), we always get start < collection_start < collection-end < end
    for row_idx in set(range(lhs_row_idx, rhs_row_idx)).difference(
        set(range(lhs_row_idx + 1, rhs_row_idx - 1))
    ):
        print(df[metric_name][row_idx])
        # rhs_row_idx won't be included because its start timestamp is no less than end_timestamp
        if (
            df["rawTimestamp"][row_idx] + collection_cycle_ns
            <= start_timestamp
        ):
            continue
        if df["rawTimestamp"][row_idx] >= end_timestamp:
            continue
        if df["rawTimestamp"][row_idx] + collection_cycle_ns >= end_timestamp:
            if df["rawTimestamp"][row_idx] <= start_timestamp:
                # collection_start < start < end < collection_end
                sm_active_ns_product += (end_timestamp - start_timestamp) * df[
                    metric_name
                ][row_idx]
            else:
                # start < collection_start < end < collection_end
                sm_active_ns_product += (
                    end_timestamp - df["rawTimestamp"][row_idx]
                ) * df[metric_name][row_idx]
        elif df["rawTimestamp"][row_idx] <= start_timestamp:
            # collection_start < start < collection_end < end
            sm_active_ns_product += (
                df["rawTimestamp"][row_idx]
                + collection_cycle_ns
                - start_timestamp
            ) * df[metric_name][row_idx]
        else:
            # start < collection_start < collection_end < end
            print(
                start_timestamp,
                df["rawTimestamp"][row_idx],
                df["rawTimestamp"][row_idx] + collection_cycle_ns,
                end_timestamp,
            )
            raise ValueError("This case should not happen")
    # TODO: should be rhs_row_idx - 1
    for row_idx in range(lhs_row_idx + 1, rhs_row_idx - 1):
        # start < collection_start < collection_end < end
        sm_active_ns_product += collection_cycle_ns * df[metric_name][row_idx]
    return sm_active_ns_product / (end_timestamp - start_timestamp)


def get_last_nvtx_range(
    filepath: str,
    pushpop_region_name: Union[str, None] = "graphiler",
    range_name: str = "my_code_range",
) -> tuple[int, int]:
    """
    Return the last nvtx range in the trace file.
    Skip the region check if pushpop_region_name is None.
    In our baseline measurement, "graphiler" is the message in nvtx.annotate() context, i.e., the pushpop_region_name.
    """
    nvtx_ranges: list[list[str]] = load_nsys_report(
        filepath, "nvtx_gpu_proj_trace", lambda x: ""  # dummy filter function
    )
    header = nvtx_ranges[0]
    print(header)
    name_idx = header.index("Name")
    # TODO: check if we need to use Orig instead of Projected
    start_idx = header.index("Projected Start (ns)")
    duration_idx = header.index("Projected Duration (ns)")
    style_idx = header.index("Style")
    last_region: Union[tuple[int, int], None] = None
    last_range_idx: int = -1
    last_range_start: int = 0
    last_range_duration: int = 0
    # Store the domain if it is the last domain
    if pushpop_region_name is not None:
        for event_idx, event in enumerate(nvtx_ranges[1:]):
            if (
                event[name_idx] == pushpop_region_name
                and event[style_idx] == "PushPop"
            ):
                if last_region is None or last_region[1] < int(
                    event[start_idx]
                ) + int(event[duration_idx]):
                    last_region = (
                        int(event[start_idx]),
                        int(event[start_idx]) + int(event[duration_idx]),
                    )
    for event_idx, event in enumerate(nvtx_ranges[1:]):
        if event[name_idx] == range_name and event[style_idx] == "StartEnd":
            # Check if the event is the last and if it is in the last domain
            tentative_last_range_idx = event_idx
            tentative_last_range_start = int(event[start_idx])
            tentative_last_range_duration = int(event[duration_idx])
            if pushpop_region_name is not None:
                if (
                    last_region is None
                    or last_region[1] < tentative_last_range_start
                    or last_region[0]
                    > tentative_last_range_start
                    + tentative_last_range_duration
                ):
                    continue
            last_range_idx = tentative_last_range_idx
            last_range_start = tentative_last_range_start
            last_range_duration = tentative_last_range_duration

    if last_range_idx == -1:
        raise ValueError(
            f"Cannot find nvtx range with name {range_name} in domain"
            f" {pushpop_region_name}"
        )
    return last_range_start, last_range_duration


# TODO: We assume n_warmups == 5, and n_epochs == 10, but we may in future get these parameters from arguments from the trace file
# TODO: If the trace file does not contain the information, we could use the default values as above
# TODO: But we need to verify the default values are correct
# After exporting to txt/json via nsys export -t json, the first type 27 event will involve argument information, e.g.,
# Type: 27
# CommEvent_ {
#   Timestamp: -290185240
#   GlobalPid: 341157691260928
#   NumOfCpus: 0
#   Command: "python"
#   WorkDir: "/home/kunww/HET/hrt"
#   Args: "-m"
#   Args: "python.HGT.train"
#   Args: "-d"
#   Args: "am"
#   Args: "--num_layers"
#   Args: "1"
#   Args: "--full_graph_training"
#   Args: "--num_classes"
#   Args: "64"
#   Args: "--n_infeat"
#   Args: "64"
#   Args: "--num_heads"
#   Args: "1"
#   NsTime: true
#   EnvironId: 17
# }


# TODO: devise algorithm to figure out the beginning id of training, i.e., after warm up ends
# training_beg_idx = len(kernel_instances)
def get_kern_sum():
    raise NotImplementedError


# TODO: implement cuda_api_trace report
def get_kern_trace_overhead(
    filepath: str,
    classify_het_kernel_func: Union[Callable[[str], str], None],
    timerange: Union[tuple[int, int], None] = None,
    API_flag: bool = False,
) -> "list[list[str]]":
    sm_metric_df: "pandas.DataFrame" = load_raw_gpu_metric_util_report(
        filepath, -1
    )
    report_name: str = "cuda_api_trace" if API_flag else "cuda_gpu_trace"
    # We use load_nsys_report rather for simplicity. If we need information from the file, we should use the per-file logic in load_from_nsys_reports_folders instead
    kern_traces: list[list[str]] = load_nsys_report(
        filepath, report_name, classify_het_kernel_func
    )
    header = kern_traces[0]
    kernel_name_idx = header.index("Name")
    start_timestamp_idx = 0
    duration_idx = 1
    if "CorrId" in header:
        corr_id_idx = header.index("CorrId")
    else:
        corr_id_idx = header.index("CorrID")
    # Make sure the unit is nanoseconds
    assert header[start_timestamp_idx] == "Start (ns)"
    assert header[duration_idx] == "Duration (ns)"
    # "Kernel name", "CorrId", "Pretty name", "Duration (ns)", "Start timestamp (ns)", optional["Kernel type", "Avg SM active"]
    kernel_instances: list[
        tuple[str, int, str, int, int, list[float | str]]
    ] = []
    for line in kern_traces[1:]:
        # Each line stores a kernel instance, i.e., launch.
        kernel_name = line[kernel_name_idx]
        if API_flag:
            pretty_name = kernel_name
        else:
            pretty_name = prettify_name_from_func_signature(kernel_name)
        duration = int(line[duration_idx])
        start_timestamp = int(line[start_timestamp_idx])
        avg_sm_active = calc_avg_sm_metrics(
            sm_metric_df, start_timestamp, start_timestamp + duration
        )
        if timerange is not None:
            if (
                start_timestamp < timerange[0]
                or start_timestamp > timerange[1]
            ):
                continue
        corr_id = int(line[corr_id_idx])
        if (
            classify_het_kernel_func is not None
            and classify_het_kernel_func(pretty_name) == "Non-HET Others"
        ):
            continue
        kernel_instance_tuple: tuple[
            str, int, str, int, int, list[float | str]
        ] = (
            kernel_name,
            corr_id,
            pretty_name,
            duration,
            start_timestamp,
            [avg_sm_active],
        )
        if classify_het_kernel_func is not None:
            kernel_instance_tuple[-1].append(
                classify_het_kernel_func(pretty_name)
            )
        kernel_instances.append(kernel_instance_tuple)

    results_csv: list[list[str]] = [
        [
            "Kernel name",
            "CorrId",
            "Pretty name",
            "Duration (ns)",
            "Start timestamp (ns)",
            "Avg SM active",
        ]
    ]
    if classify_het_kernel_func is not None:
        results_csv[0] += ["Kernel type"]
    for kernel_instance in kernel_instances:
        results_csv.append([str(cell) for cell in kernel_instance[:-1]])
        # Optional metrics, i.e., Avg SM Active and Kernel type
        results_csv[-1] += list(map(str, kernel_instance[-1]))

    return results_csv
