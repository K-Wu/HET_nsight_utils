from .run_once import run_once
from functools import lru_cache
import os
from typing import Tuple


@lru_cache(maxsize=None)
@run_once
def is_ctags_installed() -> bool:
    return os.system("ctags --version >/dev/null 2>/dev/null") == 0


def get_functions_from_ctags_table(ctags_table: str) -> set[str]:
    result = set()
    for line in ctags_table.split("\n"):
        if line.startswith("HET_") and line.endswith("f"):
            result.add(line.split("\t")[0])
    return result


def get_GEMM_kernel_names(het_root_path: str) -> "set[str]":
    assert is_ctags_installed(), "ctags is not installed"
    # HET_ functions defined in the two files are GEMM kernels.
    GEMM_files = [
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/RGNN/my_shmem_sgemm_func_rgcn_hgt.cu.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/RGNN/my_shmem_sgemm_func.cu.h",
    ]

    # Use exuberant ctags to get the function names.
    ctags_tables: list[str] = []
    for file in GEMM_files:
        assert os.path.isfile(file), f"{file} does not exist"
        ctags_tables.append(os.popen("ctags -f- " + file).read())

    return get_functions_from_ctags_table("\n".join(ctags_tables))


def get_fw_bw_host_func_names(het_root_path: str) -> Tuple[set[str], set[str]]:
    # the kernel is a forward kernel if its namespace involve "FwProp"
    # the kernel is a backward kernel if its namespace involve "BckProp"
    assert is_ctags_installed(), "ctags is not installed"
    torch_op_files = [
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/HGTOpsEdgeParallel.inc.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGCNOps.inc.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/GATOps.inc.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/HGTOps.inc.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGATOps.inc.h",
        f"{het_root_path}/hetero_edgesoftmax/include/DGLHackKernel/OpExport/RGNNOps.inc.h",
    ]

    # Use exuberant ctags to get the function names.
    ctags_tables: list[str] = []
    for file in torch_op_files:
        assert os.path.isfile(file), f"{file} does not exist"
        ctags_tables.append(os.popen("ctags -f- " + file).read())

    fw_kernels = set()
    bw_kernels = set()
    for table in ctags_tables:
        for line in table.split("\n"):
            line_components = line.split("\t")
            if len(line_components) < 3:
                print(f"WARNING: ctags table line {line} has less than 3 components")
                continue
            func_name = line_components[0]
            namespace = line_components[-1]
            tag_type = line_components[-2]
            if tag_type != "f":
                print(f"Warning: {func_name} is not a function")
                continue
            if "FwProp" in namespace:
                fw_kernels.add(func_name)
            elif "BckProp" in namespace:
                bw_kernels.add(func_name)
            else:
                print(
                    f"Warning: {func_name} is neither a forward nor a backward kernel"
                )

    return fw_kernels, bw_kernels


def classify_fw_bw_kernel(func_pretty_name: str) -> str:
    if (
        "Delta" in func_pretty_name
        or "BckProp" in func_pretty_name
        or "_bck_" in func_pretty_name
        or "Backward" in func_pretty_name
    ):
        return "BckProp"
    else:
        if (
            "FwProp" in func_pretty_name
            or "_fw_" in func_pretty_name
            or "Forward" in func_pretty_name
        ):
            return "FwProp"
        else:
            print(f"Warning: assuming {func_pretty_name} is a forward kernel")
            return "FwProp"
