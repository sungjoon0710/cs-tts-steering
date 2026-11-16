"""Small descriptive direction diagnostics; no emotion or quality classifier."""
from itertools import combinations
import numpy as np
from src.caa_stats import pca_fit


def cosine(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching finite vectors')
    denominator = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    value = np.divide(np.sum(a * b, axis=-1), denominator,
                      out=np.full(np.shape(denominator), np.nan), where=denominator > 0)
    return np.clip(value, -1, 1)


def contrasts(default, condition):
    a, b = np.asarray(default, dtype=np.float64), np.asarray(condition, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matching finite states')
    denominator = np.linalg.norm(a, axis=-1)
    if np.any(denominator == 0):
        raise ValueError('Zero default-state norm')
    return b - a, np.linalg.norm(b - a, axis=-1) / denominator


def cross_text_cosines(deltas):
    x = np.asarray(deltas, dtype=np.float64)
    if x.ndim != 3 or len(x) < 2:
        raise ValueError('Expected at least two text x layer x hidden arrays')
    return [(i, j, cosine(x[i], x[j])) for i, j in combinations(range(len(x)), 2)]


def calibration_mean(states, splits):
    x = np.asarray(states, dtype=np.float64)
    if len(x) != len(splits):
        raise ValueError('Split count mismatch')
    mask = np.asarray([s == 'calibration' for s in splits])
    if not mask.any():
        raise ValueError('No calibration examples')
    return x[mask].mean(axis=0)


def mean_direction_consistency(deltas):
    """Mean over distinct pairs, without constructing an O(n²) pair list.

    Return eligible counts; zero directions are undefined, not perfect matches.
    Pairs share texts and must not be treated as independent replications.
    """
    x = np.asarray(deltas, dtype=np.float64)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite text x layer x hidden states')
    norm = np.linalg.norm(x, axis=-1)
    count = (norm > 0).sum(axis=0)
    unit = np.divide(x, norm[..., None], out=np.zeros_like(x), where=norm[..., None] > 0)
    numerator = np.sum(unit.sum(axis=0) ** 2, axis=-1) - count
    denominator = count * (count - 1)
    value = np.divide(numerator, denominator, out=np.full(count.shape, np.nan), where=count > 1)
    return np.clip(value, -1, 1), count


def project_calibration_pca(states, splits, components=2):
    """Fit upstream PCA on calibration rows only; project all rows unchanged."""
    x = np.asarray(states, dtype=np.float64)
    mask = np.asarray([s == 'calibration' for s in splits])
    if x.ndim != 2 or len(x) != len(splits) or not np.isfinite(x).all():
        raise ValueError('Invalid PCA states/splits')
    if components < 1 or components > min(int(mask.sum()) - 1, x.shape[1]):
        raise ValueError('Insufficient calibration examples for requested PCA')
    mean, axes = pca_fit(x[mask], components)
    return {'mean': mean, 'axes': axes, 'scores': (x - mean) @ axes.T,
            'calibration_count': int(mask.sum())}
