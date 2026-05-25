"""Smoke test: the package and its key modules import and wire together."""


def test_package_imports():
    import qfr
    from qfr.data.fmp_client import FMPClient  # noqa: F401
    from qfr.data.universe import build_universe  # noqa: F401
    from qfr.utils.config import settings
    from qfr.utils.logging import logger  # noqa: F401

    assert qfr.__version__
    assert settings.fmp_base_url.startswith("http")
