"""Smoke-test the Streamlit page when local policy artifacts are available."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
DATA = (
    ROOT / "data" / "processed" / "calibration_policy" / "validation_calibration_comparison.parquet"
)


@pytest.mark.skipif(not DATA.exists(), reason="本地验证预测未生成")
def test_streamlit_dashboard_loads_without_exception() -> None:
    """The policy dashboard should load local artifacts and render its default page."""
    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert "SmartCredit｜授信策略模拟看板" in app.title[0].value
    assert len(app.metric) == 6
