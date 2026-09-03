"""Validate that SmartCredit imports from the active virtual environment."""

from smartcredit import __version__


# This minimal smoke test catches broken environments or package paths early.
def test_package_version() -> None:
    """Confirm that the installed package version matches the project definition."""
    assert __version__ == "0.1.0"
