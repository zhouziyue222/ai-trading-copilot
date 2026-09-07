"""Close UI lifespans even in legacy tests using TestClient without a context."""

import pytest


@pytest.fixture(autouse=True)
def close_test_ui_runtimes():
    from ai_trading_copilot.copilot.services.run_lifecycle import ACTIVE_RUNTIMES
    before = set(ACTIVE_RUNTIMES)
    yield
    for runtime in set(ACTIVE_RUNTIMES) - before:
        runtime.shutdown()
