import numpy as np
import pytest

from focal_stats import rolling_mean, rolling_sum, rolling_window


def test_rolling_int_window_dimensions():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)
    c = np.random.rand(10, 10, 10)
    d = np.random.rand(5, 5, 5, 5)

    assert rolling_window(a, window=5).shape == (6, 5)
    assert rolling_window(a, window=5, reduce=True).shape == (2, 5)
    assert rolling_window(a, window=5, flatten=True).shape == (6, 5)
    assert rolling_window(a, window=5, reduce=True, flatten=True).shape == (2, 5)

    assert rolling_window(b, window=5).shape == (6, 6, 5, 5)
    assert rolling_window(b, window=5, reduce=True).shape == (2, 2, 5, 5)
    assert rolling_window(b, window=5, flatten=True).shape == (6, 6, 25)
    assert rolling_window(b, window=5, reduce=True, flatten=True).shape == (2, 2, 25)

    assert rolling_window(c, window=5).shape == (6, 6, 6, 5, 5, 5)
    assert rolling_window(c, window=5, reduce=True).shape == (2, 2, 2, 5, 5, 5)
    assert rolling_window(c, window=5, flatten=True).shape == (6, 6, 6, 125)
    assert rolling_window(c, window=5, reduce=True, flatten=True).shape == (
        2,
        2,
        2,
        125,
    )

    assert rolling_window(d, window=5, reduce=True, flatten=True).shape == (
        1,
        1,
        1,
        1,
        625,
    )


def test_rolling_list_window_dimensions():
    b = np.random.rand(10, 10)
    c = np.random.rand(10, 10, 10)
    d = np.random.rand(5, 5, 5, 5)

    assert rolling_window(b, window=[5, 6]).shape == (6, 5, 5, 6)
    assert rolling_window(b, window=[5, 2], reduce=True).shape == (2, 5, 5, 2)
    assert rolling_window(b, window=[5, 6], flatten=True).shape == (6, 5, 30)
    assert rolling_window(b, window=[5, 2], reduce=True, flatten=True).shape == (
        2,
        5,
        10,
    )

    assert rolling_window(c, window=[5, 5, 6]).shape == (6, 6, 5, 5, 5, 6)
    assert rolling_window(c, window=[5, 5, 2], reduce=True).shape == (2, 2, 5, 5, 5, 2)
    assert rolling_window(c, window=[5, 5, 6], flatten=True).shape == (6, 6, 5, 150)
    assert rolling_window(c, window=[5, 5, 2], reduce=True, flatten=True).shape == (
        2,
        2,
        5,
        50,
    )

    assert rolling_window(d, window=[5, 5, 5, 5], reduce=True, flatten=True).shape == (
        1,
        1,
        1,
        1,
        625,
    )


def test_rolling_window_dimensions_mask():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)
    c = np.random.rand(10, 10, 10)
    d = np.random.rand(5, 5, 5, 5)

    s = np.random.RandomState(0)

    mask_a = s.rand(5) > 0.5
    mask_b = s.rand(5, 10) > 0.5
    mask_c = s.rand(5, 5, 10) > 0.5
    mask_d = s.rand(5, 5, 5, 5) > 0.5

    assert rolling_window(a, window=mask_a).shape == (6, mask_a.sum())
    assert rolling_window(a, window=mask_a, reduce=True).shape == (2, mask_a.sum())
    assert rolling_window(a, window=mask_a, flatten=True).shape == (6, mask_a.sum())
    assert rolling_window(a, window=mask_a, reduce=True, flatten=True).shape == (
        2,
        mask_a.sum(),
    )

    assert rolling_window(b, window=mask_b).shape == (6, 1, mask_b.sum())
    assert rolling_window(b, window=mask_b, reduce=True).shape == (2, 1, mask_b.sum())
    assert rolling_window(b, window=mask_b, flatten=True).shape == (6, 1, mask_b.sum())
    assert rolling_window(b, window=mask_b, reduce=True, flatten=True).shape == (
        2,
        1,
        mask_b.sum(),
    )

    assert rolling_window(c, window=mask_c).shape == (6, 6, 1, mask_c.sum())
    assert rolling_window(c, window=mask_c, reduce=True).shape == (
        2,
        2,
        1,
        mask_c.sum(),
    )
    assert rolling_window(c, window=mask_c, flatten=True).shape == (
        6,
        6,
        1,
        mask_c.sum(),
    )
    assert rolling_window(c, window=mask_c, reduce=True, flatten=True).shape == (
        2,
        2,
        1,
        mask_c.sum(),
    )

    assert rolling_window(d, window=mask_d, reduce=True, flatten=True).shape == (
        1,
        1,
        1,
        1,
        mask_d.sum(),
    )


def test_rolling_values():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)

    assert rolling_window(a, window=5)[0, 4] == a[4]
    assert rolling_window(a, window=5, reduce=True)[1, 4] == a[9]
    assert rolling_window(a, window=5, flatten=True)[0, 4] == a[4]
    assert rolling_window(a, window=5, reduce=True, flatten=True)[1, 4] == a[9]

    assert rolling_window(b, window=5)[2, 2, 0, 0] == b[2, 2]
    assert rolling_window(b, window=5)[2, 2, 2, 2] == b[4, 4]


def test_rolling_values_mask():
    a = np.random.rand(10)
    b = np.random.rand(10, 10)

    s = np.random.RandomState(0)

    mask_a = s.rand(5) > 0.5
    mask_b = s.rand(5, 10) > 0.5

    assert rolling_window(a, window=mask_a)[1, 0] == a[1:6][mask_a][0]
    assert rolling_window(a, window=mask_a, reduce=True)[1, 0] == a[5:][mask_a][0]
    assert rolling_window(a, window=mask_a, flatten=True)[1, 0] == a[1:6][mask_a][0]
    assert (
        rolling_window(a, window=mask_a, reduce=True, flatten=True)[1, 0]
        == a[5:][mask_a][0]
    )

    assert rolling_window(b, window=mask_b)[1, 0, 0] == b[1:6, 0:10][mask_b][0]
    assert (
        rolling_window(b, window=mask_b, reduce=True)[1, 0, 0]
        == b[5:10, 0:10][mask_b][0]
    )


def test_rolling_errors():
    a = np.random.rand(10)

    # negative window size
    with pytest.raises(ValueError):
        rolling_window(a, window=-1)

    # window size bigger than array
    with pytest.raises(ValueError):
        rolling_window(a, window=11)

    # in reduction mode the window size needs to divide the input array exactly
    with pytest.raises(ValueError):
        rolling_window(a, window=4, reduce=True)

    # no mask and no window
    with pytest.raises(ValueError):
        rolling_window(a)

    # window with wrong dimensions
    with pytest.raises(IndexError):
        rolling_window(a, window=[5, 5])

    # mask with wrong dimensions
    with pytest.raises(IndexError):
        rolling_window(a, window=[[True, True], [True, True]])


@pytest.mark.parametrize("dims", (1, 2, 3, 4))
def test_rolling_sum_int_window(dims):
    a = np.random.rand(*(10 for _ in range(dims)))
    assert np.allclose(
        rolling_sum(a, window=5), rolling_window(a, window=5, flatten=True).sum(axis=-1)
    )


@pytest.mark.parametrize("dims", (2, 3, 4))
def test_rolling_sum_list_window(dims):
    a = np.random.rand(*(10 for _ in range(dims)))
    window = [5 for _ in range(dims)]
    window[-1] = 2

    assert np.allclose(
        rolling_sum(a, window=window),
        rolling_window(a, window=window, flatten=True).sum(axis=-1),
    )


@pytest.mark.parametrize("dims", (1, 2, 3, 4))
def test_rolling_sum_mask(dims):
    s = np.random.RandomState(0)
    a = np.random.rand(*(10 for _ in range(dims)))
    mask = s.rand(*(5 for _ in range(dims))) > 0.5
    assert np.allclose(
        rolling_sum(a, window=mask), rolling_window(a, window=mask).sum(axis=-1)
    )


def test_rolling_sum_errors():
    a = np.random.rand(10, 10)

    # window 0 or lower
    with pytest.raises(ValueError):
        rolling_sum(a, window=0)

    # window bigger than input data
    with pytest.raises(ValueError):
        rolling_sum(a, window=11)


def test_rolling_mean():
    a = np.random.rand(10, 10, 10)
    assert np.allclose(rolling_mean(a, window=5)[0, 0, 0], a[:5, :5, :5].mean())
    assert np.allclose(rolling_mean(a, window=5)[-1, -1, -1], a[-5:, -5:, -5:].mean())
