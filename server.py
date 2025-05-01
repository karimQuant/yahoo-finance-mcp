import json
from enum import Enum
from typing import List # Import List
from datetime import datetime, timedelta # Import datetime and timedelta

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP
import numpy as np # Import numpy for histogram calculation


# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"


# Initialize FastMCP server
yfinance_server = FastMCP(
    "yfinance",
    instructions="""
# Yahoo Finance MCP Server

This server is used to get information about a given ticker symbol from yahoo finance.

Available tools:
- get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
- get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
- get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
- get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
- get_option_expiration_dates: Fetch the available options expiration dates for a given ticker symbol.
- get_option_chain: Fetch the option chain for a given ticker symbol, expiration date, and option type.
- get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
- get_return_distribution: Calculate the histogram distribution of returns for a stock.
- calculate_correlations: Calculate the correlation matrix for a basket of stocks over a specified time range.
""",
)


@yfinance_server.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        Default is "1mo"
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        Default is "1d"
""",
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    """Get historical stock prices for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
            Default is "1mo"
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """
    company = yf.Ticker(ticker)
    try:
        # Using info to check if ticker exists is more reliable than isin
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    # If the company is found, get the historical data
    try:
        hist_data = company.history(period=period, interval=interval)
        if hist_data.empty:
             print(f"No historical data found for ticker {ticker} with period={period}, interval={interval}.")
             return f"No historical data found for ticker {ticker} with period={period}, interval={interval}."
        hist_data = hist_data.reset_index(names="Date")
        hist_data = hist_data.to_json(orient="records", date_format="iso")
        return hist_data
    except Exception as e:
        print(f"Error: getting historical stock prices for {ticker}: {e}")
        return f"Error: getting historical stock prices for {ticker}: {e}"


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
)
async def get_stock_info(ticker: str) -> str:
    """Get stock information for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error: getting stock information for {ticker}: {e}")
        return f"Error: getting stock information for {ticker}: {e}"
    return json.dumps(info)


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    """Get news for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """
    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    # If the company is found, get the news
    try:
        news = company.news
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    news_list = []
    # Check if news is a list before iterating
    if isinstance(news, list):
        for item in news:
            # Check if the item has the expected structure
            if isinstance(item, dict) and item.get("content", {}).get("contentType", "") == "STORY":
                title = item.get("content", {}).get("title", "")
                summary = item.get("content", {}).get("summary", "")
                description = item.get("content", {}).get("description", "")
                url = item.get("content", {}).get("canonicalUrl", {}).get("url", "")
                news_list.append(
                    f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
                )
    if not news_list:
        print(f"No news found for company that searched with {ticker} ticker.")
        return f"No news found for company that searched with {ticker} ticker."
    return "\n\n".join(news_list)


@yfinance_server.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
)
async def get_stock_actions(ticker: str) -> str:
    """Get stock dividends and stock splits for a given ticker symbol"""
    try:
        company = yf.Ticker(ticker)
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    try:
        actions_df = company.actions
        if actions_df.empty:
             print(f"No stock actions found for ticker {ticker}.")
             return f"No stock actions found for ticker {ticker}."
        actions_df = actions_df.reset_index(names="Date")
        return actions_df.to_json(orient="records", date_format="iso")
    except Exception as e:
        print(f"Error: getting stock actions for {ticker}: {e}")
        return f"Error: getting stock actions for {ticker}: {e}"


@yfinance_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    """Get financial statement for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    financial_statement = None
    try:
        if financial_type == FinancialType.income_stmt:
            financial_statement = company.income_stmt
        elif financial_type == FinancialType.quarterly_income_stmt:
            financial_statement = company.quarterly_income_stmt
        elif financial_type == FinancialType.balance_sheet:
            financial_statement = company.balance_sheet
        elif financial_type == FinancialType.quarterly_balance_sheet:
            financial_statement = company.quarterly_balance_sheet
        elif financial_type == FinancialType.cashflow:
            financial_statement = company.cashflow
        elif financial_type == FinancialType.quarterly_cashflow:
            financial_statement = company.quarterly_cashflow
        else:
            return f"Error: invalid financial type {financial_type}. Please use one of the following: {FinancialType.income_stmt.value}, {FinancialType.quarterly_income_stmt.value}, {FinancialType.balance_sheet.value}, {FinancialType.quarterly_balance_sheet.value}, {FinancialType.cashflow.value}, {FinancialType.quarterly_cashflow.value}."

        if financial_statement is None or financial_statement.empty:
             print(f"No {financial_type} found for ticker {ticker}.")
             return f"No {financial_type} found for ticker {ticker}."

    except Exception as e:
        print(f"Error: getting financial statement for {ticker}: {e}")
        return f"Error: getting financial statement for {ticker}: {e}"


    # Create a list to store all the json objects
    result = []

    # Loop through each column (date)
    for column in financial_statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")  # Format as YYYY-MM-DD
        else:
            date_str = str(column)

        # Create a dictionary for each date
        date_obj = {"date": date_str}

        # Add each metric as a key-value pair
        for index, value in financial_statement[column].items():
            # Add the value, handling NaN values
            date_obj[index] = None if pd.isna(value) else value

        result.append(date_obj)

    return json.dumps(result)


@yfinance_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    try:
        if holder_type == HolderType.major_holders:
            holders_df = company.major_holders
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.reset_index(names="metric").to_json(orient="records")
        elif holder_type == HolderType.institutional_holders:
            holders_df = company.institutional_holders
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.to_json(orient="records")
        elif holder_type == HolderType.mutualfund_holders:
            holders_df = company.mutualfund_holders
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_transactions:
            holders_df = company.insider_transactions
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_purchases:
            holders_df = company.insider_purchases
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.to_json(orient="records", date_format="iso")
        elif holder_type == HolderType.insider_roster_holders:
            holders_df = company.insider_roster_holders
            if holders_df.empty:
                 print(f"No {holder_type} found for ticker {ticker}.")
                 return f"No {holder_type} found for ticker {ticker}."
            return holders_df.to_json(orient="records", date_format="iso")
        else:
            return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders.value}, {HolderType.institutional_holders.value}, {HolderType.mutualfund_holders.value}, {HolderType.insider_transactions.value}, {HolderType.insider_purchases.value}, {HolderType.insider_roster_holders.value}."
    except Exception as e:
        print(f"Error: getting holder info for {ticker}: {e}")
        return f"Error: getting holder info for {ticker}: {e}"


@yfinance_server.tool(
    name="get_option_expiration_dates",
    description="""Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    """Fetch the available options expiration dates for a given ticker symbol."""

    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    try:
        options = company.options
        if not options:
             print(f"No option expiration dates found for ticker {ticker}.")
             return f"No option expiration dates found for ticker {ticker}."
        return json.dumps(options)
    except Exception as e:
        print(f"Error: getting option expiration dates for {ticker}: {e}")
        return f"Error: getting option expiration dates for {ticker}: {e}"


@yfinance_server.tool(
    name="get_option_chain",
    description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.

Args:
    ticker: str
        The ticker symbol of the stock to get option chain for, e.g. "AAPL"
    expiration_date: str
        The expiration date for the options chain (format: 'YYYY-MM-DD')
    option_type: str
        The type of option to fetch ('calls' or 'puts')
""",
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    """Fetch the option chain for a given ticker symbol, expiration date, and option type.

    Args:
        ticker: The ticker symbol of the stock
        expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: The type of option to fetch ('calls' or 'puts')

    Returns:
        str: JSON string containing the option chain data
    """

    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    try:
        # Check if the expiration date is valid
        if expiration_date not in company.options:
            return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."

        # Check if the option type is valid
        if option_type not in ["calls", "puts"]:
            return "Error: Invalid option type. Please use 'calls' or 'puts'."

        # Get the option chain
        option_chain = company.option_chain(expiration_date)
        if option_type == "calls":
            chain_df = option_chain.calls
        elif option_type == "puts":
            chain_df = option_chain.puts
        else: # This case should not be reached due to the check above, but included for safety
             return f"Error: invalid option type {option_type}. Please use one of the following: calls, puts."

        if chain_df.empty:
             print(f"No {option_type} option chain found for ticker {ticker} on date {expiration_date}.")
             return f"No {option_type} option chain found for ticker {ticker} on date {expiration_date}."

        return chain_df.to_json(orient="records", date_format="iso")

    except Exception as e:
        print(f"Error: getting option chain for {ticker} on date {expiration_date}: {e}")
        return f"Error: getting option chain for {ticker} on date {expiration_date}: {e}"


@yfinance_server.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"
    try:
        if recommendation_type == RecommendationType.recommendations:
            recs_df = company.recommendations
            if recs_df.empty:
                 print(f"No recommendations found for ticker {ticker}.")
                 return f"No recommendations found for ticker {ticker}."
            return recs_df.to_json(orient="records")
        elif recommendation_type == RecommendationType.upgrades_downgrades:
            upgrades_downgrades = company.upgrades_downgrades
            if upgrades_downgrades.empty:
                 print(f"No upgrades/downgrades found for ticker {ticker}.")
                 return f"No upgrades/downgrades found for ticker {ticker}."
            # Get the upgrades/downgrades based on the cutoff date
            upgrades_downgrades = upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            if upgrades_downgrades.empty:
                 print(f"No upgrades/downgrades found for ticker {ticker} in the last {months_back} months.")
                 return f"No upgrades/downgrades found for ticker {ticker} in the last {months_back} months."
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            # Get the first occurrence (most recent) for each firm
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
        else:
             return f"Error: invalid recommendation type {recommendation_type}. Please use one of the following: {RecommendationType.recommendations.value}, {RecommendationType.upgrades_downgrades.value}."
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"


@yfinance_server.tool(
    name="get_return_distribution",
    description="""Calculate the histogram distribution of returns for a stock.

Args:
    ticker: str
        The ticker symbol of the stock, e.g. "AAPL"
    return_period_days: int
        The number of days over which each individual return is calculated.
    days_back: int
        The total number of trading days of historical returns to consider for the distribution.
    number_of_bins: int
        The number of bins for the histogram.
""",
)
async def get_return_distribution(
    ticker: str, return_period_days: int, days_back: int, number_of_bins: int
) -> str:
    """Calculate the histogram distribution of returns for a stock."""
    company = yf.Ticker(ticker)
    try:
        info = company.info
        if not info:
             print(f"Company ticker {ticker} not found or no info available.")
             return f"Company ticker {ticker} not found or no info available."
    except Exception as e:
        print(f"Error checking ticker {ticker}: {e}")
        return f"Error checking ticker {ticker}: {e}"

    # Fetch historical data. Fetching 'max' period is safest to ensure enough data.
    try:
        hist_data = company.history(period="max")
        if hist_data.empty:
             print(f"No historical data found for ticker {ticker}.")
             return f"No historical data found for ticker {ticker}."
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return f"Error fetching historical data for {ticker}: {e}"

    # Calculate the rolling return over return_period_days
    # Use pct_change with periods to get the return over the specified number of days
    returns = hist_data['Close'].pct_change(periods=return_period_days)

    # Drop initial NaN values resulting from the rolling calculation
    returns = returns.dropna()

    # Ensure we have enough data points after dropping NaNs
    if len(returns) < days_back:
         print(f"Not enough historical data to calculate {days_back} returns over {return_period_days} days for {ticker}. Found {len(returns)} valid returns.")
         return f"Not enough historical data to calculate {days_back} returns over {return_period_days} days for {ticker}. Found {len(returns)} valid returns."

    # Take the last 'days_back' returns
    recent_returns = returns.tail(days_back)

    # Calculate the histogram
    try:
        counts, bin_edges = np.histogram(recent_returns, bins=number_of_bins)
    except Exception as e:
        print(f"Error calculating histogram for {ticker}: {e}")
        return f"Error calculating histogram for {ticker}: {e}"

    # Format the result
    # Convert numpy arrays to lists for JSON serialization
    histogram_data = {
        "counts": counts.tolist(),
        "bin_edges": bin_edges.tolist()
    }

    return json.dumps(histogram_data)


@yfinance_server.tool(
    name="calculate_correlations",
    description="""Calculate the correlation matrix for a basket of stocks over a specified time range.
Fetches historical daily prices for the given tickers and calculates the correlation matrix of their daily returns.

Args:
    tickers: List[str]
        A list of ticker symbols, e.g., ["AAPL", "MSFT", "GOOG"]
    days_back: int
        The number of historical trading days to consider for the calculation.
""",
)
async def calculate_correlations(tickers: List[str], days_back: int) -> str:
    """Calculate the correlation matrix for a basket of stocks."""

    if not tickers:
        return "Error: Please provide a list of ticker symbols."

    if days_back <= 0:
        return "Error: days_back must be a positive integer."

    # Calculate the start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back * 1.5) # Fetch a bit more data to account for non-trading days

    try:
        # Fetch historical data for all tickers
        # yf.download returns a dict of dataframes or a single dataframe if tickers is a list
        data = yf.download(tickers, start=start_date, end=end_date)

        if data.empty:
             print(f"No historical data found for tickers {tickers} in the last {days_back} days.")
             return f"No historical data found for tickers {tickers} in the last {days_back} days."

        # Extract 'Close' prices for all tickers
        # If only one ticker, data is a single DataFrame, need to handle this
        if len(tickers) == 1:
             close_prices = data['Close'].to_frame() # Convert Series to DataFrame
        else:
             close_prices = data['Close']

        # Drop columns where all values are NaN (e.g., ticker not found or no data)
        close_prices = close_prices.dropna(axis=1, how='all')

        if close_prices.empty:
             print(f"No valid historical close price data found for any of the tickers {tickers}.")
             return f"No valid historical close price data found for any of the tickers {tickers}."

        # Check if any tickers were dropped
        dropped_tickers = set(tickers) - set(close_prices.columns)
        if dropped_tickers:
             print(f"Could not retrieve data for tickers: {', '.join(dropped_tickers)}")
             # Decide if you want to return an error or proceed with available tickers
             # For now, proceed with available tickers but warn the user
             # return f"Error: Could not retrieve data for tickers: {', '.join(dropped_tickers)}"
             pass # Proceed with available data

        # Calculate daily returns
        returns = close_prices.pct_change().dropna()

        if returns.empty:
             print(f"Not enough data to calculate returns for tickers {close_prices.columns} over the last {days_back} days.")
             return f"Not enough data to calculate returns for tickers {close_prices.columns} over the last {days_back} days."

        # Ensure we have at least 2 data points for correlation calculation
        if len(returns) < 2:
             print(f"Not enough data points ({len(returns)}) to calculate correlation for tickers {close_prices.columns} over the last {days_back} days. Need at least 2.")
             return f"Not enough data points ({len(returns)}) to calculate correlation for tickers {close_prices.columns} over the last {days_back} days. Need at least 2."


        # Calculate the correlation matrix
        correlation_matrix = returns.corr()

        # Convert the correlation matrix DataFrame to JSON
        # orient='index' or 'columns' works well for matrix-like data
        # We'll use 'index' which makes the index (tickers) the outer keys
        correlation_json = correlation_matrix.to_json(orient='index')

        return correlation_json

    except Exception as e:
        print(f"Error calculating correlations for tickers {tickers}: {e}")
        return f"Error calculating correlations for tickers {tickers}: {e}"


if __name__ == "__main__":
    # Initialize and run the server
    print("Starting Yahoo Finance MCP server...")
    yfinance_server.run(transport="stdio")
