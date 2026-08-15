"""Shared fixtures. The synthetic dataset is built once per session; it is deterministic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader, features  # noqa: E402


@pytest.fixture(scope="session")
def dataset():
    return data_loader.load()


@pytest.fixture(scope="session")
def panel(dataset):
    return dataset.daily


@pytest.fixture(scope="session")
def feature_frame(dataset):
    return features.build_full_features(dataset.daily, dataset.cot, dataset.macro_vintages)
