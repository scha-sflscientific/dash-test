"""
This module contains an example test.

Tests should be placed in ``tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``pytest --cov -p no:warnings``.
"""
from pathlib import Path

import pytest

from src.predict import predict
from src import config, logger


@pytest.fixture
def project_config():
    return config


class TestInferencePipeline:

    """Inference Pipeline Integration Test"""

    def test_inference_config_section(self, project_config):
        """Test necessary config sections"""
        assert config.get("DATA") != None
        assert config.get("INFERENCE") != None

    def test_inference_pipeline(self):
        """Inference Pipeline Integration Test"""
        predict()
