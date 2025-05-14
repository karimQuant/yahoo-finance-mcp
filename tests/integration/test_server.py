import pytest
import json
from server import (
    get_historical_stock_prices,
    get_stock_info,
    get_yahoo_finance_news,
    get_stock_actions,
    get_financial_statement,
    get_holder_info,
    get_option_expiration_dates,
    get_option_chain,
    get_recommendations,
    get_return_distribution,
    calculate_correlations,
    FinancialType,
    HolderType,
    RecommendationType,
)

# Use a common ticker for testing
TEST_TICKER = "AAPL"
# Use a basket of tickers for correlation test
TEST_TICKERS_BASKET = ["AAPL", "MSFT", "GOOG"]


def is_valid_response(response: str) -> bool:
    """Checks if the response is not an error message and is not empty."""
    return isinstance(response, str) and not response.startswith("Error:") and response.strip() != ""

def is_valid_json_response(response: str) -> bool:
    """Checks if the response is valid JSON and not an error message."""
    if not is_valid_response(response):
        return False
    try:
        json.loads(response)
        return True
    except json.JSONDecodeError:
        return False


@pytest.mark.asyncio
async def test_get_historical_stock_prices():
    """Test get_historical_stock_prices tool."""
    response = await get_historical_stock_prices(ticker=TEST_TICKER, period="5d", interval="1d")
    assert is_valid_json_response(response)
    # Optional: Add more specific checks for JSON structure if needed


@pytest.mark.asyncio
async def test_get_stock_info():
    """Test get_stock_info tool."""
    response = await get_stock_info(ticker=TEST_TICKER)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_get_yahoo_finance_news():
    """Test get_yahoo_finance_news tool."""
    response = await get_yahoo_finance_news(ticker=TEST_TICKER)
    assert is_valid_response(response)


@pytest.mark.asyncio
async def test_get_stock_actions():
    """Test get_stock_actions tool."""
    response = await get_stock_actions(ticker=TEST_TICKER)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_get_financial_statement():
    """Test get_financial_statement tool (e.g., income_stmt)."""
    response = await get_financial_statement(ticker=TEST_TICKER, financial_type=FinancialType.income_stmt.value)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_get_holder_info():
    """Test get_holder_info tool (e.g., major_holders)."""
    response = await get_holder_info(ticker=TEST_TICKER, holder_type=HolderType.major_holders.value)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_get_option_expiration_dates():
    """Test get_option_expiration_dates tool."""
    response = await get_option_expiration_dates(ticker=TEST_TICKER)
    assert is_valid_json_response(response)
    # Optional: Check if the JSON is a list of strings


@pytest.mark.asyncio
async def test_get_option_chain():
    """Test get_option_chain tool."""
    # First, get a valid expiration date
    dates_response = await get_option_expiration_dates(ticker=TEST_TICKER)
    assert is_valid_json_response(dates_response)
    expiration_dates = json.loads(dates_response)

    assert expiration_dates, f"No expiration dates found for {TEST_TICKER}"

    # Use the first available date
    expiration_date = expiration_dates[0]

    # Test calls
    calls_response = await get_option_chain(ticker=TEST_TICKER, expiration_date=expiration_date, option_type="calls")
    assert is_valid_json_response(calls_response)

    # Test puts
    puts_response = await get_option_chain(ticker=TEST_TICKER, expiration_date=expiration_date, option_type="puts")
    assert is_valid_json_response(puts_response)


@pytest.mark.asyncio
async def test_get_recommendations():
    """Test get_recommendations tool (e.g., recommendations)."""
    response = await get_recommendations(ticker=TEST_TICKER, recommendation_type=RecommendationType.recommendations.value)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_get_return_distribution():
    """Test get_return_distribution tool."""
    response = await get_return_distribution(ticker=TEST_TICKER, return_period_days=1, days_back=100, number_of_bins=10)
    assert is_valid_json_response(response)


@pytest.mark.asyncio
async def test_calculate_correlations():
    """Test calculate_correlations tool."""
    response = await calculate_correlations(tickers=TEST_TICKERS_BASKET, days_back=30)
    assert is_valid_json_response(response)
    # Optional: Check if the JSON contains the expected tickers as keys/columns
