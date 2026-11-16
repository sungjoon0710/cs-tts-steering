import numpy as np
import pytest
from src.representation_geometry import (calibration_mean, contrasts, cosine,
    cross_text_cosines, mean_direction_consistency, project_calibration_pca)


def test_known_directions_and_undefined_zero():
    assert cosine([1, 0], [0, 1]) == 0
    assert cosine([1, 0], [-2, 0]) == -1
    assert np.isnan(cosine([0, 0], [1, 0]))
    delta, ratio = contrasts([[3, 4]], [[6, 8]])
    np.testing.assert_array_equal(delta, [[3, 4]])
    np.testing.assert_allclose(ratio, [1])
    with pytest.raises(ValueError):
        contrasts([[0, 0]], [[1, 0]])
    with pytest.raises(ValueError):
        cosine([np.nan, 1], [1, 1])


def test_fast_consistency_matches_distinct_pairs_and_excludes_zeros():
    x = np.array([[[1., 0]], [[2., 0]], [[0., 1]], [[0., -1]]])
    explicit = np.stack([v for i, j, v in cross_text_cosines(x)])
    fast, n = mean_direction_consistency(x)
    assert len(explicit) == 6 and n.tolist() == [4]
    np.testing.assert_allclose(fast, explicit.mean(axis=0))
    assert np.isnan(mean_direction_consistency(np.zeros((3, 1, 2)))[0][0])


def test_calibration_mean_and_pca_do_not_see_heldout_values():
    x = np.array([[1., 0, 2], [0, 2, 1], [2, 2, 0], [10000, 10000, 10000]])
    splits = ['calibration'] * 3 + ['heldout']
    mean = calibration_mean(x, splits)
    first = project_calibration_pca(x, splits)
    x[-1] *= -1000
    np.testing.assert_array_equal(calibration_mean(x, splits), mean)
    second = project_calibration_pca(x, splits)
    np.testing.assert_array_equal(first['axes'], second['axes'])
    np.testing.assert_array_equal(first['mean'], second['mean'])
    assert not np.array_equal(first['scores'][-1], second['scores'][-1])
