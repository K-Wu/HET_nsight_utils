from typing import Callable
from .load_nsight_report import (
    prettify_name_from_func_signature,
    load_nsys_report,
)
from typing import Union


def get_last_nvtx_range(
    filepath: str,
    domain_name: Union[str, None] = "graphiler",
    range_name: str = "my_code_range",
) -> tuple[int, int]:
    """
    Return the last nvtx range in the trace file.
    Skip the domain check if domain_name is None.
    """
    nvtx_ranges: list[list[str]] = load_nsys_report(
        filepath, "nvtx_gpu_proj_trace", lambda x: ""  # dummy filter function
    )
    header = nvtx_ranges[0]
    name_idx = header.index("Name")
    # TODO: check if we need to use Orig instead of Projected
    start_idx = header.index("Projected Start (ns)")
    duration_idx = header.index("Projected Duration (ns)")
    style_idx = header.index("Style")
    last_domain: Union[tuple[int, int], None] = None
    last_range_idx: int = -1
    last_range_start: int = 0
    last_range_duration: int = 0
    # Store the domain if it is the last domain
    if domain_name is not None:
        for event_idx, event in enumerate(nvtx_ranges[1:]):
            if (
                event[name_idx] == domain_name
                and event[style_idx] == "PushPop"
            ):
                if last_domain is None or last_domain[1] < int(
                    event[start_idx]
                ) + int(event[duration_idx]):
                    last_domain = (
                        int(event[start_idx]),
                        int(event[start_idx]) + int(event[duration_idx]),
                    )
    for event_idx, event in enumerate(nvtx_ranges[1:]):
        if event[name_idx] == range_name and event[style_idx] == "StartEnd":
            # Check if the event is the last and if it is in the last domain
            tentative_last_range_idx = event_idx
            tentative_last_range_start = int(event[start_idx])
            tentative_last_range_duration = int(event[duration_idx])
            if domain_name is not None:
                if (
                    last_domain is None
                    or last_domain[1] < tentative_last_range_start
                    or last_domain[0]
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
            f" {domain_name}"
        )
    return last_range_start, last_range_duration


# TODO: we assume n_warmups == 5, and n_epochs == 10, but we may in future get these parameters from arguments from the trace file
"""
After exporting to txt/json via nsys export -t json, the first type 27 event will involve argument information, e.g.,
Type: 27
CommEvent_ {
  Timestamp: -290185240
  GlobalPid: 341157691260928
  NumOfCpus: 0
  Command: "python"
  WorkDir: "/home/kunww/HET/hrt"
  Args: "-m"
  Args: "python.HGT.train"
  Args: "-d"
  Args: "am"
  Args: "--num_layers"
  Args: "1"
  Args: "--full_graph_training"
  Args: "--num_classes"
  Args: "64"
  Args: "--n_infeat"
  Args: "64"
  Args: "--num_heads"
  Args: "1"
  NsTime: true
  EnvironId: 17
}
"""


# TODO: devise algorithm to figure out the beginning id of training, i.e., after warm up ends
# training_beg_idx = len(kernel_instances)
def get_kern_sum(
    filepath: str,
    classify_het_kernel_func: Union[Callable[[str], str], None],
    timerange: Union[tuple[int, int], None] = None,
    API_flag: bool = False,
) -> "list[list[str]]":
    report_name: str = "cuda_api_trace" if API_flag else "cuda_gpu_trace"
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
    kernel_instances: list[
        Union[
            tuple[str, int, str, int, int, str], tuple[str, int, str, int, int]
        ]
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
        kernel_instance_tuple = (
            kernel_name,
            corr_id,
            pretty_name,
            duration,
            start_timestamp,
        )
        if classify_het_kernel_func is not None:
            kernel_instance_tuple += (classify_het_kernel_func(pretty_name),)
        kernel_instances.append(kernel_instance_tuple)

    results_csv: list[list[str]] = [
        [
            "Kernel name",
            "CorrId",
            "Pretty name",
            "Duration (ns)",
            "Start timestamp (ns)",
        ]
    ]
    if classify_het_kernel_func is not None:
        results_csv[0] += ["Kernel type"]
    results_csv += [list(map(str, row)) for row in kernel_instances]
    return results_csv


# TODO: implement cuda_api_trace report
