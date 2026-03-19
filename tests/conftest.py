import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked as slow (requires downloading transformer models)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow (skipped unless --runslow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="Pass --runslow to run model-loading tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
