if __name__ == "__main__":
    from .. import load_nsys_report as general_load_nsys_report
    from .. import consolidate_ncu_details as general_consolidate_ncu_details
    from .. import (
        extract_ncu_values_from_details,
        reorder_columns_in_raw_csv,
        load_ncu_report,
        extract_ncu_values_from_raws,
        calculate_roofline_for_ncu_raw_csvs,
    )
    import os
    from .utils import is_pwd_correct_for_testing, get_path_to_test_dir

    assert is_pwd_correct_for_testing(), (
        "Please run this script at the directory where the nsight_utils is in."
        " The command will be something like python -m"
        " nsight_utils.test.test_load_nsys_report"
    )

    # Hard code GEMM kernel names for test purpose
    GEMM_kernels = {
        "HET_RGNNDeltaNodeFeatInputCompactBckProp",
        "HET_RGNNDeltaWeightBckPropACGatherScatterListIdentical",
        "HET_RGCNMatmulNoScatterGatherListDeltaWeightBckProp",
        "HET_HGTFusedAttnScoreDeltaKVectBckProp",
        "HET_RGNNDeltaWeightBckProp",
        "HET_RGNNMatmulNoScatterGatherListFwProp",
        "HET_HGTMessageGenerationAndAccumulationDeltaWeightBckProp",
        "HET_RGNNDeltaNodeFeatInputBckPropACGatherScatterListIdentical",
        "HET_RGNNDeltaWeightCompactBckProp",
        "HET_RGNNDeltaWeightNoScatterGatherListBckProp",
        "HET_RGNNDeltaNodeFeatInputBckProp",
        "HET_RGNNMatmulNoScatterGatherDeltaFeatBckProp",
        "HET_HGTFusedAttnScoreDeltaWeightBckProp",
        "HET_RGNNFeatPerEdgeFwProp",
        "HET_HGTFusedAttnScoreFwProp",
        "HET_RGNNFeatPerEdgeFwPropACGatherScatterListIdentical",
        "HET_RGCNMatmulNoScatterGatherListDeltaNodeFeatBckProp",
        "HET_RGNNFeatCompactFwProp",
        "HET_HGTMessageGenerationAndAccumulationDeltaNodeFeatInputBckProp",
        "HET_HGTMessageGenerationAndAccumulationFwProp",
        "HET_RGCNMatmulNoScatterGatherListFwProp",
    }

    # Define helper functions for test purpose
    def classify_het_kernel(func_name: str) -> str:
        if func_name in GEMM_kernels:
            return "GEMM"
        elif func_name.startswith("HET_"):
            return "Traversal"
        else:
            return "Non-HET Others"

    def load_nsys_report(filename: str, report_name: str):
        return general_load_nsys_report(
            filename, report_name, classify_het_kernel
        )

    def consolidate_ncu_details(
        metric_per_row: "list[list[str]]",
    ) -> "list[list[str]]":
        return general_consolidate_ncu_details(
            metric_per_row, classify_het_kernel
        )

    print(
        consolidate_ncu_details(
            extract_ncu_values_from_details(
                load_ncu_report(
                    os.path.join(
                        get_path_to_test_dir(), "HGT.aifb...64.64.1.ncu-rep"
                    ),
                    "details",
                )
            )
        )
    )

    print(
        reorder_columns_in_raw_csv(
            extract_ncu_values_from_raws(
                load_ncu_report(
                    os.path.join(
                        get_path_to_test_dir(), "HGT.aifb...64.64.1.ncu-rep"
                    ),
                    "raw",
                )
            ),
            metric_front=[],
            metric_end=[],
        )
    )

    print(
        calculate_roofline_for_ncu_raw_csvs(
            extract_ncu_values_from_raws(
                load_ncu_report(
                    os.path.join(
                        get_path_to_test_dir(), "HGT.aifb...64.64.1.ncu-rep"
                    ),
                    "raw",
                )
            )
        )[0:2]
    )

    print(
        load_nsys_report(
            os.path.join(
                get_path_to_test_dir(), "graphiler_hgt_fb15k.nsys-rep"
            ),
            "cuda_gpu_trace",
        )
    )

    print(
        load_nsys_report(
            os.path.join(
                get_path_to_test_dir(), "graphiler_hgt_fb15k.nsys-rep"
            ),
            "cuda_gpu_trace,nvtx_sum,osrt_sum,cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum",
        )
    )
