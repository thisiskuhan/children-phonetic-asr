"""Shared pytest configuration — warning filters for external libraries."""

import pytest


def pytest_configure(config):
    """Suppress known third-party deprecation warnings that we can't fix."""
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:torchaudio",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore::FutureWarning:torchaudio",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore::UserWarning:torchaudio",
    )
    # torch checkpoint reentrant warning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*use_reentrant.*:FutureWarning:torch",
    )
    # torch attention mask mismatch warning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:.*key_padding_mask.*:UserWarning:torch",
    )
