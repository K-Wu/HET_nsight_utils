"""Adapted from /opt/nvidia/nsight-systems/2023.3.1/target-linux-x64/python/packages/nsys_recipe/gpu_metric_util_map/gpu_metric_util_map.py
SmActive, SmIssue, and TensorActive are all percentages
rawTimestamp is in nanoseconds
"""

from .load_nsight_report import (
    get_nsys_recipe_package_path,
    get_nsysstats_package_path,
    extract_sqlite_from_nsys_report,
)
import sys

sys.path.append(get_nsys_recipe_package_path())
sys.path.append(get_nsysstats_package_path())
import nsysstats
from nsys_recipe.lib import helpers
import argparse
import pandas

class RawGpuMetricUtilReport(nsysstats.Report):
    """The rawTimestamp is the beginning timestamp of the collected metric. In other words, in nsys, the timeline [rawTimestamp, rawTimestamp + 1.0sec/metricCollectionFrequency] shows the metric value at rawTimestamp."""

    query_metrics = """
WITH
    metrics AS (
        SELECT
            rawTimestamp AS rawTimestamp,
            typeId & 0xFF AS gpu,
            LEAD (rawTimestamp) OVER (PARTITION BY typeId) end,
            CAST(JSON_EXTRACT(data, '$.SM Active') as INT) AS smActive,
            CAST(JSON_EXTRACT(data, '$.SM Issue') as INT) AS smIssue,
            CAST(JSON_EXTRACT(data, '$.Tensor Active') as INT) AS tensorActive
        FROM
            GENERIC_EVENTS
    )
SELECT
    rawTimestamp AS "rawTimestamp",
    smActive AS "SmActive",
    smIssue AS "SmIssue",
    tensorActive AS "TensorActive",
    gpu AS "GPU"
FROM
    metrics
"""

    table_checks = {
        "ANALYSIS_DETAILS": "{DBFILE} does not contain analysis details.",
        "GENERIC_EVENTS": "{DBFILE} does not contain GPU metric data.",
    }

    def setup(self):
        err = super().setup()
        if err != None:
            return err

        self.statements = []

        if self.parsed_args.rows > 0:
            print(
                "Limiting query to {ROW_LIMIT} rows".format(
                    ROW_LIMIT=self.parsed_args.rows
                )
            )
            self.query_metrics += "LIMIT {ROW_LIMIT}".format(
                ROW_LIMIT=self.parsed_args.rows
            )
        else:
            print("Querying all rows")

        self.query = self.query_metrics


def load_raw_gpu_metric_util_report(
    nsys_filename: str, row_limit: int = -1
) -> "pandas.DataFrame":
    """Load the raw gpu metric util report from the nsys report file. row_limit is by default -1, meaning loading all rows."""
    parsed_args = argparse.Namespace(
        rows=row_limit,
    )
    extract_sqlite_from_nsys_report(nsys_filename)
    assert nsys_filename.endswith(".nsys-rep")
    db_filename = nsys_filename[: nsys_filename.rfind(".nsys-rep")] + ".sqlite"
    return helpers.stats_cls_to_df(
        db_filename, parsed_args, RawGpuMetricUtilReport
    )


if __name__ == "__main__":
    parsed_args = argparse.Namespace(
        rows=-1,
    )
    extract_sqlite_from_nsys_report(
        "utils/nsight_utils/test/graphiler.fb15k_RGAT.bg.breakdown.nsys-rep"
    )
    df = helpers.stats_cls_to_df(
        "utils/nsight_utils/test/graphiler.fb15k_RGAT.bg.breakdown.sqlite",
        parsed_args,
        RawGpuMetricUtilReport,
    )
    print(df[df["SmActive"] > 0])
