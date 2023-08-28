import pytest
from rasterio.windows import Window

from focal_stats.windows import define_windows, WindowPair


def test_window_definition_errors():
    # Not divisible by window_size
    with pytest.raises(ValueError):
        define_windows((10, 10), window_size=3, reduce=True)

    # Too large
    with pytest.raises(ValueError):
        define_windows((1, 1), window_size=2, reduce=False)

    # Too small
    with pytest.raises(ValueError):
        define_windows((1, 1), window_size=0, reduce=False)


def test_window_definition_reduce():
    assert (
            list(define_windows((5, 5), window_size=5, reduce=True)) ==
            [WindowPair(Window(0, 0, 5, 5), Window(0, 0, 1, 1))]
    )


def test_window_definition_non_reduce():
    for wp in list(define_windows((2, 2), window_size=1, reduce=False)):
        assert (
                wp in
                [
                    WindowPair(input=Window(col_off=0, row_off=0, width=1, height=1),
                               output=Window(col_off=0, row_off=0, width=1, height=1)),
                    WindowPair(input=Window(col_off=1, row_off=0, width=1, height=1),
                               output=Window(col_off=1, row_off=0, width=1, height=1)),
                    WindowPair(input=Window(col_off=0, row_off=1, width=1, height=1),
                               output=Window(col_off=0, row_off=1, width=1, height=1)),
                    WindowPair(input=Window(col_off=1, row_off=1, width=1, height=1),
                               output=Window(col_off=1, row_off=1, width=1, height=1))
                ]
        )