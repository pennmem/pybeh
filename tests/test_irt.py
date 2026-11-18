"""Tests for pybeh.irt.irt.

Each expected value is worked by hand. With recall times t1 < t2 < ... on a trial, irts[i, j] is
t(j+2) - t(j+1): the time from recall j + 1 to recall j + 2, and 0 where there is no next recall.

Run from the repository root:  python -m pytest
"""
import numpy as np
import pytest

from pybeh.irt import irt


def test_last_interval_of_a_full_row():
    # Four recalls in a matrix four columns wide. Before the fix the third entry came back as 7000, the
    # raw time of the third recall, instead of the interval 15000 - 7000.
    out = irt([[1000, 3000, 7000, 15000]])
    np.testing.assert_array_equal(out, [[2000, 4000, 8000, 0]])


def test_row_with_a_single_padding_zero():
    # Two recalls in a matrix three columns wide. Before the fix the second entry came back as 9, the
    # raw time of the last recall.
    out = irt([[1, 2, 3], [5, 9, 0]])
    np.testing.assert_array_equal(out, [[1, 1, 0], [4, 0, 0]])


def test_zero_padded_rows():
    times = [[1000, 3000, 7000, 15000, 0, 0],
             [500, 900, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]
    expected = [[2000, 4000, 8000, 0, 0, 0],
                [400, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]]
    np.testing.assert_array_equal(irt(times), expected)


def test_single_recall_has_no_interval():
    np.testing.assert_array_equal(irt([[2500, 0, 0]]), [[0, 0, 0]])


def test_nan_padding_is_treated_as_no_recall():
    times = np.array([[1000., 3000., np.nan], [500., np.nan, np.nan]])
    np.testing.assert_array_equal(irt(times), [[2000, 0, 0], [0, 0, 0]])


def test_output_has_the_shape_of_the_input():
    times = np.zeros((7, 5))
    assert irt(times).shape == (7, 5)
    assert irt(np.zeros((0, 4))).shape == (0, 4)


@pytest.mark.parametrize('make', [lambda x: [list(r) for r in x], np.array],
                         ids=['list of lists', 'numpy array'])
def test_input_is_not_modified(make):
    original = [[1000, 3000, 7000, 15000], [500, 900, 0, 0]]
    times = make(original)
    irt(times)
    np.testing.assert_array_equal(np.array(times), np.array(original))


def test_random_trials_match_differences_of_recall_times():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n_cols = int(rng.integers(1, 9))
        n = int(rng.integers(0, n_cols + 1))          # number of recalls on the trial
        t = np.sort(rng.choice(np.arange(1, 75000), size=n, replace=False))
        row = np.zeros(n_cols)
        row[:n] = t
        out = irt(row[None, :])[0]
        if n > 1:
            np.testing.assert_array_equal(out[:n - 1], np.diff(t))
        assert not out[max(n - 1, 0):].any()
