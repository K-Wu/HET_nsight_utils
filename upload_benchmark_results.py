# Some code is from https://github.com/COVID19Tracking/ltc-data-processing
# And https://github.com/nlioc4/FSBot/blob/f7f1a000ec7d02056c136fe68b7f0ca2271c80ae/modules/accounts_handler.py#L326
# To create a credential, or set up a new spreasheet, follow instruction at https://docs.gspread.org/en/latest/oauth2.html#for-bots-using-service-account
import gspread
from gspread import Worksheet, WorksheetNotFound
from gspread.utils import finditem
from typing import Union, Any
from abc import abstractmethod

import os
import socket


class ConfigCanonicalizer:
    @classmethod
    def permute(cls, input_fmt: str, config: "list[str]") -> "list[str]":
        """
        sort the config list according to reverse-alphabetical order of input_fmt
        """
        input_fmt_: list[str] = input_fmt.split(".")
        assert len(input_fmt_) == len(config)
        return [
            c
            for _, c in sorted(
                zip(input_fmt_, config), key=lambda pair: pair[0], reverse=True
            )
        ]

    # Define your own validation rule here in subclasses
    @classmethod
    @abstractmethod
    def validate_config_fmt(cls, input_fmt: str) -> None:
        configs = input_fmt.split(".")
        if "ax_in" not in configs:
            print("ax_in is not in configs")
        if "ax_out" not in configs:
            print("ax_out is not in configs")
        if "ax_head" not in configs:
            print("ax_head is not in configs")

    @classmethod
    def canonicalize_list(
        cls,
        config: "list[str]",
        input_fmt: str,
        prettifier_rule: "dict[str,str]" = {
            "multiply_among_weights_first_flag": "Fusion",
            "compact_as_of_node_flag--compact_direct_indexing_flag": (
                "CompactDirect"
            ),
            "compact_as_of_node_flag": "Compact",
            "": "None",  # Replace null values to allow joining in Tableau
        },
    ) -> "list[str]":
        """
        Example of input_fmt: "flag_mul.flag_compact.ax_in.ax_out.ax_head"
        """
        cls.validate_config_fmt(input_fmt)
        if input_fmt is not None:
            config = cls.permute(input_fmt, config)
        ret: list[str] = [c[2:] if c.startswith("--") else c for c in config]
        ret = [prettifier_rule[c] if c in prettifier_rule else c for c in ret]
        return ret

    # Define your own get_dimensions here in subclasses
    @classmethod
    @abstractmethod
    def get_dimensions(cls, config: "list[str]", input_fmt: str) -> str:
        input_fmts = input_fmt.split(".")
        ax_in_idx = input_fmts.index("ax_in")
        ax_out_idx = input_fmts.index("ax_out")
        ax_head_idx = input_fmts.index("ax_head")
        return (
            f"{config[ax_in_idx]}.{config[ax_out_idx]}.{config[ax_head_idx]}"
        )

    # Define your own get_configs_other_than_dimensions here in subclasses
    @classmethod
    @abstractmethod
    def get_configs_other_than_dimensions(
        cls, config: "list[str]", input_fmt: str
    ) -> str:
        config = cls.canonicalize_list(config, input_fmt)
        input_fmts = input_fmt.split(".")
        ax_in_idx = input_fmts.index("ax_in")
        ax_out_idx = input_fmts.index("ax_out")
        ax_head_idx = input_fmts.index("ax_head")
        other_configs: "list[str]" = [
            c
            for idx, c in enumerate(config)
            if idx not in {ax_in_idx, ax_out_idx, ax_head_idx}
        ]
        if max([len(c) for c in other_configs]) == 0:
            # Use $UNOPT to represent the unoptimized config
            return "$UNOPT"
        else:
            return ".".join(other_configs)

    @classmethod
    def to_str(cls, config: "list[str]", input_fmt: str) -> str:
        return ".".join(cls.canonicalize_list(config, input_fmt))


class NameCanonicalizer:
    @classmethod
    def to_list(cls, name: str, input_fmt: str) -> "list[str]":
        input_fmt_: list[str] = input_fmt.split(".")
        name_: list[str] = name.split(".")
        assert len(input_fmt_) == len(name_)
        config_fmt = ".".join(
            [ele for ele in input_fmt_ if ele not in {"model", "dataset"}]
        )
        model = name_[input_fmt_.index("model")]
        dataset = name_[input_fmt_.index("dataset")]
        configs = [
            name_[idx]
            for idx in range(len(name_))
            if input_fmt_[idx] not in {"model", "dataset"}
        ]
        return [model, dataset] + ConfigCanonicalizer.canonicalize_list(
            configs, config_fmt
        )

    @classmethod
    def to_str(cls, name: str, input_fmt: str) -> str:
        return ".".join(cls.to_list(name, input_fmt))


def get_pretty_hostname() -> str:
    # kwu-csl227-99-CEntosREfugee will be kwu-csl227-99
    hostname = socket.gethostname()
    hostname_parts = hostname.split("-")
    if len(hostname_parts) > 2:
        return (
            hostname_parts[0]
            + "-"
            + hostname_parts[1]
            + "-"
            + hostname_parts[2]
        )
    else:
        return hostname


def count_rows(csv_rows: "list[list[Any]]") -> int:
    return len(csv_rows)


def count_cols(csv_rows: "list[list[Any]]") -> int:
    return max([len(row) for row in csv_rows])


def find_latest_subdirectory_or_file(
    dirname: str, prefix: str, suffix: str = ""
) -> str:
    candidates: list[str] = []
    for subdir_or_file in os.listdir(dirname):
        if subdir_or_file.startswith(prefix) and subdir_or_file.endswith(
            suffix
        ):
            candidates.append(subdir_or_file)
    return os.path.join(dirname, max(candidates))


def ask_subdirectory_or_file(
    dirname, prefix, results_relative_dir, suffix=""
) -> str:
    """
    Show latest directory and request user input
    If user input is empty then choose the latest directory
    otherwise, choose the user input
    """
    candidate = find_latest_subdirectory_or_file(dirname, prefix, suffix)
    print(
        "With prefix ",
        prefix,
        " and suffix ",
        suffix,
        (
            ", the latest directory/file (depending on which type you"
            " request) is "
        ),
        os.path.basename(candidate),
    )
    user_input = input(
        "Press enter to use it, or please input the directory (without prefix"
        " or suffix) you want to use (e.g, upload):"
    )
    if len(user_input) == 0:
        result = candidate
    else:
        if user_input.startswith(
            "///"
        ):  # user input is a relative path to het root
            assert user_input[3:].startswith(results_relative_dir)
            user_input = os.path.relpath(user_input[3:], results_relative_dir)
        result = os.path.join(dirname, user_input)
    assert os.path.exists(result), f"{result} does not exist"
    return result


def ask_subdirectory(
    dirname: str, prefix: str, results_relative_dir: str
) -> str:
    while 1:
        result = ask_subdirectory_or_file(
            dirname, prefix, results_relative_dir
        )
        if os.path.isdir(result):
            return result
        else:
            print(result, "is not a directory. try again")
    raise RuntimeError("Unreachable")


def ask_file(dirname: str, prefix: str, suffix: str = "") -> str:
    while 1:
        result = ask_subdirectory_or_file(dirname, prefix, "", suffix)
        if os.path.isfile(result):
            return result
        else:
            print(result, "is not a file. try again")
    raise RuntimeError("Unreachable")


def generate_filename(
    dirname: str, prefix: str, suffix: str, time_format="%Y%m%d%H%M"
) -> str:
    """
    Generate a filename with prefix and timestamp
    """
    import time

    ret: str = prefix + time.strftime(time_format, time.localtime()) + suffix
    if os.path.exists(os.path.join(dirname, ret)):
        print("WARNING: ", ret, "already exists")
    return ret


def open_worksheet(target_sheet_url: str, target_gid: str):
    if target_gid != "0":
        raise NotImplementedError(
            "To avoid data loss, only gid=0 is supported for now"
        )
    gc = gspread.service_account()
    sh = gc.open_by_url(target_sheet_url)
    sheet_data = sh.fetch_sheet_metadata()

    try:
        item = finditem(
            lambda x: str(x["properties"]["sheetId"]) == target_gid,
            sheet_data["sheets"],
        )
        ws = Worksheet(sh, item["properties"])
    except (StopIteration, KeyError):
        raise WorksheetNotFound(target_gid)
    return ws


def create_worksheet(
    target_sheet_url: str, title: str, retry=False
) -> Worksheet:
    gc = gspread.service_account()
    sh = gc.open_by_url(target_sheet_url)
    title_suffix = ""
    # when retry is True, we will ask user to specify a suffix if the title already exists
    if retry:
        while True:
            if (title + title_suffix)[:100] in [
                ws.title for ws in sh.worksheets()
            ]:
                # ask user to specify a suffix
                title_suffix = input(
                    "title already exists, please specify a suffix:"
                )
            else:
                break

    return sh.add_worksheet(title=title + title_suffix, rows=100, cols=20)


def get_cell_range_from_A1(
    num_rows: int, num_cols: int, row_idx_beg: int = 0, col_idx_beg: int = 0
) -> str:
    """
    In future, we may use a1_range_to_grid_range to get the boundary of an existent worksheet.
    a1_range_to_grid_range returns (beg, end] for both row and column, i.e.,
    a1_range_to_grid_range('A1:A1')
    {'startRowIndex': 0, 'endRowIndex': 1, 'startColumnIndex': 0, 'endColumnIndex': 1}
    """
    # rowcol_to_a1(1,1) == 'A1'
    cell_range = gspread.utils.rowcol_to_a1(row_idx_beg + 1, col_idx_beg + 1)
    cell_range += ":"
    cell_range += gspread.utils.rowcol_to_a1(
        row_idx_beg + num_rows, col_idx_beg + num_cols
    )
    print(cell_range)
    return cell_range


def try_best_to_numeric(
    csv_rows: "list[list[Union[float, str, int]]]",
) -> "list[list[Union[float, str, int]]]":
    new_csv_rows: "list[list[Union[float, str, int]]]" = []
    for row in csv_rows:
        new_row = []
        for ele in row:
            if isinstance(ele, str) and ele.isnumeric():
                new_row.append(int(ele))
            elif isinstance(ele, str) and ele.replace(".", "", 1).isnumeric():
                new_row.append(float(ele))
            else:
                new_row.append(ele)
        new_csv_rows.append(new_row)
    return new_csv_rows


def update_gspread(entries, ws: Worksheet, cell_range=None) -> None:
    if cell_range is None:
        # start from A1
        num_rows = len(entries)
        num_cols = max([len(row) for row in entries])
        cell_range = get_cell_range_from_A1(num_rows, num_cols)
    ws.format(
        cell_range, {"numberFormat": {"type": "NUMBER", "pattern": "0.0000"}}
    )
    ws.update(cell_range, try_best_to_numeric(entries))
    # ws.update_title("[GID0]TestTitle")

    # Format example:
    # cells_list = ws.range(1, 1, num_rows, num_cols) # row, column, row_end, column_end. 1 1 stands for A1
    # cells_list = ws.range("E1:G120")
    # ws.format(cell_range, {"numberFormat": {"type": "DATE", "pattern": "mmmm dd"}, "horizontalAlignment": "CENTER"})
