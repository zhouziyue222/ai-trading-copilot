from ai_trading_copilot.copilot.adapters import finnhub


def test_finnhub_news_formatter_includes_published_time_and_url():
    output = finnhub._format_news_item_with_metadata(
        {
            "headline": "Memory Stocks valuation disagreement widens",
            "source": "MarketWatch",
            "summary": "Earnings will decide the next move.",
            "url": "https://example.com/memory-stocks",
            "datetime": 1786881600,
        }
    )

    assert "published_at: " in output
    assert "source: MarketWatch" in output
    assert "title: Memory Stocks valuation disagreement widens" in output
    assert "url: https://example.com/memory-stocks" in output
    assert "summary: Earnings will decide the next move." in output


def test_finnhub_news_formatter_marks_missing_time_and_url_unknown():
    output = finnhub._format_news_item_with_metadata(
        {
            "headline": "Untimed article",
            "source": "Finnhub",
        }
    )

    assert "published_at: unknown" in output
    assert "url: unknown" in output
    assert "summary: unknown" in output
