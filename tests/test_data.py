"""Validate deterministic cleaning rules for modeling data."""

import numpy as np
import pandas as pd

from smartcredit.data import clean_features


def test_clean_features_normalizes_sentinel_infinity_and_category_missing() -> None:
    """Normalize employment sentinels, infinities, and missing categories for models."""
    frame = pd.DataFrame(
        {
            "DAYS_EMPLOYED": [365243, -1000],
            "AMT_CREDIT": [np.inf, 100000],
            "NAME_INCOME_TYPE": [None, "Working"],
        }
    )

    result = clean_features(
        frame,
        numeric_columns=["DAYS_EMPLOYED", "AMT_CREDIT"],
        categorical_columns=["NAME_INCOME_TYPE"],
    )

    assert pd.isna(result.loc[0, "DAYS_EMPLOYED"])
    assert pd.isna(result.loc[0, "AMT_CREDIT"])
    assert result["DAYS_EMPLOYED"].dtype == np.dtype("float32")
    assert result.loc[0, "NAME_INCOME_TYPE"] == "__MISSING__"
