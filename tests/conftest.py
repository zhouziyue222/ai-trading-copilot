"""Close UI lifespans even in legacy tests using TestClient without a context."""

import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def isolate_default_delayed_reviews(tmp_path, monkeypatch):
    """Legacy UI tests must not schedule the user's real pending review jobs."""
    from ai_trading_copilot.copilot import review, run
    from ai_trading_copilot.copilot.ui import app
    from ai_trading_copilot.copilot.services import delayed_review
    factory = delayed_review.create_review_service

    def isolated(path):
        if Path(path).resolve() == run.DEFAULT_MEMORY_DATABASE.resolve():
            path = tmp_path / "default-reviews.sqlite3"
        return factory(path)

    monkeypatch.setenv("COPILOT_REVIEW_ACCOUNTS", "[]")
    for module in (review, app, delayed_review):
        monkeypatch.setattr(module, "create_review_service", isolated)


@pytest.fixture(autouse=True)
def close_test_ui_runtimes():
    from ai_trading_copilot.copilot.services.run_lifecycle import ACTIVE_RUNTIMES
    before = set(ACTIVE_RUNTIMES)
    yield
    for runtime in set(ACTIVE_RUNTIMES) - before:
        runtime.shutdown()
