"""
Utility helpers for ETF performance dashboard.

Modules are split into configuration, data fetching, and storage helpers.
"""

from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Return package version if installed via pip, fallback to dev."""
    try:
        return version("etf_dashboard")
    except PackageNotFoundError:
        return "0.1.dev"


__all__ = ["get_version"]
