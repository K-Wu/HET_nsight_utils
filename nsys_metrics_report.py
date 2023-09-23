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


import logging

LOG = logging.getLogger(__name__)

logging.basicConfig(
    format=(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d]"
        " %(threadName)15s: %(message)s"
    ),
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

sys.path.append(get_nsys_recipe_package_path())
sys.path.append(get_nsysstats_package_path())
import nsysstats
from nsys_recipe.lib import helpers
import argparse


class RawGpuMetricUtilReport(nsysstats.Report):
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
            LOG.info(
                "Limiting query to {ROW_LIMIT} rows".format(
                    ROW_LIMIT=self.parsed_args.rows
                )
            )
            self.query_metrics += "LIMIT {ROW_LIMIT}".format(
                ROW_LIMIT=self.parsed_args.rows
            )
        else:
            LOG.info("Querying all rows")

        self.query = self.query_metrics


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
