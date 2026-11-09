"""Black solid fills written by openpyxl carry ARGB '00000000' and must load as 1.0.

Excel writes black as 'FF000000'; openpyxl and pandas write it as '00000000'.
The loader previously treated '00000000' as "no fill" and returned 0.0, so an
openpyxl checkerboard loaded as all zeros and was rejected.
"""

import pytest
from openpyxl import Workbook
from openpyxl.styles import PatternFill

import excel_patterns



def _checkerboard(path, size=8):
    wb = Workbook()
    ws = wb.active
    black = PatternFill("solid", start_color="000000", end_color="000000")
    white = PatternFill("solid", start_color="FFFFFF", end_color="FFFFFF")
    for i in range(size):
        for j in range(size):
            ws.cell(row=i + 1, column=j + 1).fill = black if (i + j) % 2 == 0 else white
    wb.save(path)
    wb.close()


def test_openpyxl_black_checkerboard_loads_black_as_one_and_white_as_zero(tmp_path):
    path = str(tmp_path / "checker.xlsx")
    _checkerboard(path)

    pattern = excel_patterns.ExcelPatternLoader.load_pattern_from_excel(path)

    assert pattern.shape == (8, 8)
    for i in range(8):
        for j in range(8):
            expected = 1.0 if (i + j) % 2 == 0 else 0.0
            assert pattern[i, j] == pytest.approx(expected), (i, j)


def test_missing_colour_is_still_no_stimulation():
    assert excel_patterns.ExcelPatternLoader.rgb_to_intensity(None) == 0.0


def test_excel_style_ff_alpha_greys_map_by_luminance(tmp_path):
    """Fills as Excel itself writes them (FF alpha) are unaffected by the fix."""
    path = str(tmp_path / "greys.xlsx")
    wb = Workbook()
    ws = wb.active
    for j, argb in enumerate(("FF000000", "FF808080", "FFC0C0C0", "FFFFFFFF")):
        ws.cell(row=1, column=j + 1).fill = PatternFill("solid", start_color=argb, end_color=argb)
    wb.save(path)
    pattern = excel_patterns.ExcelPatternLoader.load_pattern_from_excel(path)
    assert pattern[0, 0] == pytest.approx(1.0)
    assert pattern[0, 1] == pytest.approx(1 - 128 / 255)
    assert pattern[0, 2] == pytest.approx(1 - 192 / 255)


def test_unreadable_colour_raises_rather_than_reading_as_blank():
    with pytest.raises(ValueError):
        excel_patterns.ExcelPatternLoader.rgb_to_intensity("FFZZ0000")
    with pytest.raises(ValueError):
        excel_patterns.ExcelPatternLoader.rgb_to_intensity("FFF")
