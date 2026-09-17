"""Opt in to hardware-backed tests while keeping configuration checks local."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-tpu", action="store_true", help="Run tests that start real TPU servers"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "tpu: starts TPU servers and runs evaluations")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-tpu"):
        return
    skip = pytest.mark.skip(reason="Use --run-tpu to run TPU integration tests")
    for item in items:
        if item.get_closest_marker("tpu"):
            item.add_marker(skip)
