import pytest

from focal_stats.raster_window import RasterWindow, RasterWindowPair, construct_windows


def test_window_definition_errors():
    with pytest.raises(ValueError):
        construct_windows((1, 1), window_shape=(0, 0), reduce=False)


def test_window_definition_reduce():
    assert list(construct_windows((5, 5), window_shape=(5, 5), reduce=True)) == [
        RasterWindowPair(
            input=RasterWindow(col_off=0, row_off=0, width=5, height=5),
            output=RasterWindow(col_off=0, row_off=0, width=1, height=1),
        )
    ]


def test_window_definition_non_reduce():
    for wp in list(construct_windows((2, 2), window_shape=(1, 1), reduce=False)):
        assert wp in [
            RasterWindowPair(
                input=RasterWindow(col_off=0, row_off=0, width=1, height=1),
                output=RasterWindow(col_off=0, row_off=0, width=1, height=1),
            ),
            RasterWindowPair(
                input=RasterWindow(col_off=1, row_off=0, width=1, height=1),
                output=RasterWindow(col_off=1, row_off=0, width=1, height=1),
            ),
            RasterWindowPair(
                input=RasterWindow(col_off=0, row_off=1, width=1, height=1),
                output=RasterWindow(col_off=0, row_off=1, width=1, height=1),
            ),
            RasterWindowPair(
                input=RasterWindow(col_off=1, row_off=1, width=1, height=1),
                output=RasterWindow(col_off=1, row_off=1, width=1, height=1),
            ),
        ]
