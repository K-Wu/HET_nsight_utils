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


def canonicalize_unit(unit: str) -> tuple[int, "list[str]", "list[str]"]:
    # extract exponential, numerator, denominator from unit
    if len(unit.split("/")) > 1:
        numerator = canonicalize_unit(unit.split("/")[0])
        denominator = canonicalize_unit(unit.split("/")[1])
        return _div_two_units(numerator, denominator)
    if unit[0] in UNITS_TO_EXPONENTIAL:
        exponential = UNITS_TO_EXPONENTIAL[unit[0]]
        return exponential, [unit[1:]], []
    else:
        return 0, [unit], []


def _simplify_unit_fraction(
    nominator: "list[str]", denominator: "list[str]"
) -> tuple["list[str]", "list[str]"]:
    # simplify the fraction
    for idx in range(len(nominator)):
        if nominator[idx] in denominator:
            denominator[denominator.index(nominator[idx])] = ""
            nominator[idx] = ""
    nominator = [ele for ele in nominator if len(ele) > 0]
    denominator = [ele for ele in denominator if len(ele) > 0]
    return nominator, denominator


def _div_two_units(
    lhs: tuple[int, "list[str]", "list[str]"],
    rhs: tuple[int, "list[str]", "list[str]"],
) -> tuple[int, "list[str]", "list[str]"]:
    nominator = lhs[1] + rhs[2]
    denominator = rhs[1] + lhs[2]
    nominator, denominator = _simplify_unit_fraction(nominator, denominator)
    return lhs[0] - rhs[0], nominator, denominator


def _mul_two_units(
    lhs: tuple[int, "list[str]", "list[str]"],
    rhs: tuple[int, "list[str]", "list[str]"],
) -> tuple[int, "list[str]", "list[str]"]:
    nominator = lhs[1] + rhs[1]
    denominator = rhs[2] + lhs[2]
    nominator, denominator = _simplify_unit_fraction(nominator, denominator)
    return lhs[0] + rhs[0], nominator, denominator


def div_two_units(lhs: str, rhs: str) -> str:
    # div_two_units("Tbyte", "cycle/nsecond") = "Kbyte/cycle"
    return unit_to_str(
        *_div_two_units(canonicalize_unit(lhs), canonicalize_unit(rhs))
    )


if __name__ == "__main__":
    assert mul_two_units("cycle/nsecond", "Kbyte/cycle") == "Tbyte/second"
    assert div_two_units("cycle/nsecond", "cycle/nsecond") == ""
    assert div_two_units("cycle/nsecond", "cycle") == "G/second"
    assert div_two_units("cycle", "cycle/nsecond") == "nsecond"
    assert div_two_units("cycle", "cycle") == ""
    assert div_two_units("cycle", "nsecond") == "Gcycle/second"
    assert div_two_units("nsecond", "cycle") == "nsecond/cycle"
    assert div_two_units("nsecond", "nsecond") == ""
    assert mul_two_units("nsecond", "cycle/nsecond") == "cycle"
    assert mul_two_units("cycle/nsecond", "nsecond") == "cycle"
    assert div_two_units("cycle/nsecond", "cycle") == "G/second"
    if False:
        # TODO: Not supported: unit_to_str() currently only support at most 1 nominator and 1 denominator
        print(div_two_units("Tbyte", "cycle/nsecond"))
        print(div_two_units("cycle/nsecond", "Tbyte"))
