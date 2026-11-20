"""CPU checks for calibration and document-level summaries in M3."""

import numpy as np

from scripts.m3.run_mechanism import mean_updates
from scripts.m3.summarize_mechanism import bootstrap


def test_mean_updates_average_calibration_differences_by_prediction_row():
    baseline = np.array([[1.0, 2.0], [3.0, 6.0]])
    calibration = [
        {"news": baseline, "anger": np.array([[1.0, 2.0], [3.0, 6.0]])},
        {"news": baseline, "anger": np.array([[3.0, 3.0], [4.0, 6.0]])},
    ]

    updates = mean_updates(calibration, ["anger"], [28, 30])

    np.testing.assert_array_equal(updates["anger", 28], [1.0, 0.5])
    np.testing.assert_array_equal(updates["anger", 30], [0.5, 0.0])
    np.testing.assert_array_equal(updates["anger", "global"], [0.75, 0.25])


def test_bootstrap_keeps_missing_values_out_of_the_denominator():
    count, mean, low, high = bootstrap([np.nan, 2.0, 2.0])

    assert (count, mean, low, high) == (2, 2.0, 2.0, 2.0)
