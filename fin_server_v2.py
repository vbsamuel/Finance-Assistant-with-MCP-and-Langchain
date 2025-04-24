# fin_server_v2.py
import os
import httpx
from typing import Any, Optional, List, Dict, Tuple
from pydantic import Field, BaseModel, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import asyncio
import logging
from datetime import date, timedelta # Import date utilities

# --- Imports from FastMCP ---
from fastmcp import FastMCP, Context
# --- ---

# --- Basic Logging Setup (Server Side) ---
# Configure logging for the server script itself
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use __name__ for logger name
# --- ---

# --- Configuration ---
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    finnhub_api_key: Optional[str] = Field(None, validation_alias='FINNHUB_API_KEY')
    alpha_vantage_api_key: Optional[str] = Field(None, validation_alias='ALPHA_VANTAGE_API_KEY')

settings = Settings()

# Initial checks and warnings
if not settings.finnhub_api_key:
    log.warning("FINNHUB_API_KEY not found in .env or environment. News features will be limited.")
if not settings.alpha_vantage_api_key:
    log.warning("ALPHA_VANTAGE_API_KEY not found in .env or environment. Stock price fallback and market movers will not work.")

# --- FastMCP Server Setup ---
mcp = FastMCP(
    name="FinanceDataServerV2", # Updated name slightly for clarity
    instructions="Provides tools and resources to get stock prices, market news, and top market movers."
    # Dependencies list removed - add via --with flags if using `fastmcp install`
)

# --- Pydantic Models ---
class StockQuote(BaseModel):
    ticker: str
    price: float
    change: float
    percent_change: float
    day_high: float
    day_low: float
    day_open: float
    previous_close: float
    source: str
    timestamp: Optional[int] = None

class NewsArticle(BaseModel):
    category: str
    datetime: int
    headline: str
    id: int
    image: str
    related: str
    source: str
    summary: str
    url: str

class MarketMover(BaseModel):
    ticker: str
    price: str
    change_amount: str
    change_percentage: str
    volume: str

class MarketMoversData(BaseModel):
    metadata: Optional[str] = None
    last_updated: Optional[str] = None
    top_gainers: List[MarketMover] = Field(default_factory=list)
    top_losers: List[MarketMover] = Field(default_factory=list)
    most_actively_traded: List[MarketMover] = Field(default_factory=list)


# --- API Client ---
http_client = httpx.AsyncClient(timeout=15.0)

# --- Tools ---
@mcp.tool()
async def get_stock_price(ticker: str, ctx: Context) -> Dict[str, Any]:
    """
    Fetches the current stock price quote for a given ticker symbol.
    Tries Finnhub first, then Alpha Vantage as a fallback.
    Returns a dictionary with quote data or an error message under the 'error' key.
    """
    ticker = ticker.upper()
    await ctx.info(f"Tool Call: get_stock_price for {ticker}")

    # Try Finnhub First
    if settings.finnhub_api_key:
        await ctx.debug(f"Attempting Finnhub price for {ticker}")
        try:
            response = await http_client.get(
                "https://finnhub.io/api/v1/quote",
                params={"symbol": ticker, "token": settings.finnhub_api_key}
            )
            response.raise_for_status()
            data = response.json()
            if data and data.get('c') is not None and data.get('c') != 0:
                await ctx.info(f"Success from Finnhub for {ticker}")
                quote = StockQuote(ticker=ticker, price=data.get('c', 0.0), change=data.get('d', 0.0), percent_change=data.get('dp', 0.0), day_high=data.get('h', 0.0), day_low=data.get('l', 0.0), day_open=data.get('o', 0.0), previous_close=data.get('pc', 0.0), timestamp=data.get('t'), source="Finnhub")
                return quote.model_dump()
            else: await ctx.debug(f"Finnhub returned zero/no price for {ticker}. Data: {data}")
        except httpx.HTTPStatusError as e: await ctx.warning(f"Finnhub HTTP error for {ticker}: {e.response.status_code}")
        except Exception as e: await ctx.warning(f"Error fetching {ticker} from Finnhub: {type(e).__name__} - {e}")

    # Fallback to Alpha Vantage
    if settings.alpha_vantage_api_key:
        await ctx.debug(f"Attempting Alpha Vantage price for {ticker}")
        try:
            params = {"function": "GLOBAL_QUOTE", "symbol": ticker, "apikey": settings.alpha_vantage_api_key}
            response = await http_client.get("https://www.alphavantage.co/query", params=params)
            response.raise_for_status()
            data = response.json()
            quote_data = data.get("Global Quote")
            if quote_data and quote_data.get("05. price") and quote_data.get("05. price") not in ["0.0000", "0.0"]:
                await ctx.info(f"Success from Alpha Vantage for {ticker}")
                quote = StockQuote(ticker=ticker, price=float(quote_data.get('05. price', 0.0)), change=float(quote_data.get('09. change', 0.0)), percent_change=float(quote_data.get('10. change percent', '0%').rstrip('%')), day_high=float(quote_data.get('03. high', 0.0)), day_low=float(quote_data.get('04. low', 0.0)), day_open=float(quote_data.get('02. open', 0.0)), previous_close=float(quote_data.get('08. previous close', 0.0)), source="Alpha Vantage")
                return quote.model_dump()
            elif quote_data is not None:
                 await ctx.warning(f"Alpha Vantage incomplete/zero price for {ticker}. Response: {data}")
                 return {"error": f"Incomplete or zero price data from Alpha Vantage for {ticker}", "ticker": ticker}
            else:
                 await ctx.warning(f"Alpha Vantage no 'Global Quote' data for {ticker}. Response: {data}")
                 if "Error Message" in data: return {"error": f"Alpha Vantage API Error: {data['Error Message']}", "ticker": ticker}
                 elif "Information" in data: return {"error": f"Alpha Vantage API Info: {data['Information']}", "ticker": ticker}
        except httpx.HTTPStatusError as e: await ctx.warning(f"Alpha Vantage HTTP error for {ticker}: {e.response.status_code}")
        except Exception as e: await ctx.warning(f"Error fetching {ticker} from Alpha Vantage: {type(e).__name__} - {e}")

    # If both fail
    await ctx.error(f"Failed to fetch stock price for {ticker} from all sources.")
    return {"error": f"Could not fetch valid price quote for {ticker} from available sources.", "ticker": ticker}

# Correctly registered as a TOOL
@mcp.tool()
async def get_market_movers(ctx: Context, limit_per_category: int = 5) -> Dict[str, Any]:
    """
    Gets the lists of top gaining, top losing, and most actively traded stocks
    for the current or latest trading day from Alpha Vantage.
    Returns a dictionary with keys 'top_gainers', 'top_losers', 'most_actively_traded',
    'metadata', 'last_updated', or an 'error' key if retrieval fails.
    """
    print("DEBUG: Server received call for get_market_movers TOOL") # Added debug print
    if not settings.alpha_vantage_api_key:
        log.error("Alpha Vantage API key not configured for market movers tool.")
        await ctx.error("Alpha Vantage API key is not configured.")
        return {"error": "Alpha Vantage API key not configured"}

    await ctx.info(f"Tool Call: get_market_movers (limit: {limit_per_category})")
    try:
        params = {"function": "TOP_GAINERS_LOSERS", "apikey": settings.alpha_vantage_api_key}
        response = await http_client.get("https://www.alphavantage.co/query", params=params)
        response.raise_for_status()
        data = response.json()

        if not data or ("Note" in data or "Information" in data):
            note = data.get("Note") if data else "Empty response"
            if not note and data: note = data.get("Information", "Unknown API message")
            await ctx.warning(f"Alpha Vantage API Note/Limit/Empty response: {note}")
            error_msg = f"Alpha Vantage API Error/Note: {note}"
            if data and "call frequency" in note.lower(): error_msg = f"Alpha Vantage API rate limit likely hit: {note}"
            elif not data: error_msg = "Received empty response from Alpha Vantage market movers API."
            return {"error": error_msg}

        output_data = {}
        try:
            if "top_gainers" not in data or "top_losers" not in data or "most_actively_traded" not in data:
                 await ctx.warning(f"Alpha Vantage response missing expected keys (gainers/losers/active). Raw: {data}")
                 return {"error": "Received incomplete data structure from market movers API."}
            movers = MarketMoversData.model_validate(data)
            output_data["top_gainers"] = [g.model_dump() for g in movers.top_gainers[:limit_per_category]]
            output_data["top_losers"] = [l.model_dump() for l in movers.top_losers[:limit_per_category]]
            output_data["most_actively_traded"] = [a.model_dump() for a in movers.most_actively_traded[:limit_per_category]]
            output_data["metadata"] = movers.metadata
            output_data["last_updated"] = movers.last_updated
            await ctx.info(f"Success fetching market movers. Gainers: {len(output_data['top_gainers'])}, Losers: {len(output_data['top_losers'])}, Active: {len(output_data['most_actively_traded'])}")
        except Exception as pydantic_error:
            await ctx.error(f"Failed to parse Alpha Vantage market movers response: {pydantic_error}. Raw data: {data}", exc_info=True)
            return {"error": "Failed to parse market movers data from API"}
        return output_data

    except httpx.HTTPStatusError as e:
        await ctx.error(f"Alpha Vantage HTTP error fetching market movers: {e.response.status_code} - {e.response.text}")
        return {"error": f"Alpha Vantage API error: {e.response.status_code}"}
    except Exception as e:
        await ctx.error(f"Unexpected error fetching market movers: {e}", exc_info=True)
        return {"error": "Failed to fetch market movers"}


# --- Resources ---

@mcp.resource(uri="news://market/general", description="Provides general market news headlines.")
async def get_general_market_news() -> List[Dict[str, Any]]: # No ctx
    """Fetches general market news headlines from Finnhub."""
    if not settings.finnhub_api_key:
        log.error("Finnhub API key not configured for general news resource.")
        return [{"error": "Finnhub API key not configured"}]
    log.info("Resource Call: Fetching general market news from Finnhub")
    try:
        params = {"category": "general", "token": settings.finnhub_api_key}
        response = await http_client.get("https://finnhub.io/api/v1/news", params=params)
        response.raise_for_status()
        news_list = response.json()
        if not isinstance(news_list, list):
             log.warning(f"Finnhub news response was not a list: {news_list}")
             return [{"error": "Unexpected news format from API"}]
        validated_news = []
        for article_data in news_list[:10]:
             try:
                 article = NewsArticle(**article_data)
                 validated_news.append(article.model_dump())
             except Exception as val_err:
                 log.warning(f"Skipping news article due to validation error: {val_err} - Data: {article_data}")
        log.info(f"Fetched {len(validated_news)} general news articles.")
        return validated_news
    except httpx.HTTPStatusError as e:
        log.error(f"Finnhub HTTP error fetching general news: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Finnhub API error: {e.response.status_code}"}]
    except Exception as e:
        log.exception("Error fetching general news from Finnhub")
        return [{"error": "Failed to fetch general news"}]

@mcp.resource(uri="news://ticker/{ticker}", description="Provides news headlines for a specific stock ticker.")
async def get_ticker_news(ticker: str) -> List[Dict[str, Any]]: # No ctx
    """Fetches news headlines for a specific ticker from Finnhub."""
    ticker = ticker.upper()
    if not settings.finnhub_api_key:
        log.error(f"Finnhub API key not configured for ticker news ({ticker}).")
        return [{"error": "Finnhub API key not configured"}]
    log.info(f"Resource Call: Fetching news for ticker {ticker} from Finnhub")
    try:
        today = date.today()
        one_week_ago = today - timedelta(days=7)
        params = {"symbol": ticker, "from": one_week_ago.isoformat(), "to": today.isoformat(), "token": settings.finnhub_api_key}
        response = await http_client.get("https://finnhub.io/api/v1/company-news", params=params)
        response.raise_for_status()
        news_list = response.json()
        if not isinstance(news_list, list):
             log.warning(f"Finnhub news response for {ticker} was not a list: {news_list}")
             return [{"error": f"Unexpected news format for {ticker} from API"}]
        validated_news = []
        for article_data in news_list[:10]:
             try:
                 article = NewsArticle(**article_data)
                 validated_news.append(article.model_dump())
             except Exception as val_err:
                  log.warning(f"Skipping news article for {ticker} due to validation error: {val_err} - Data: {article_data}")
        log.info(f"Fetched {len(validated_news)} news articles for {ticker}.")
        return validated_news
    except httpx.HTTPStatusError as e:
        log.error(f"Finnhub HTTP error fetching news for {ticker}: {e.response.status_code} - {e.response.text}")
        return [{"error": f"Finnhub API error for {ticker}: {e.response.status_code}"}]
    except Exception as e:
        log.exception(f"Error fetching news for {ticker} from Finnhub")
        return [{"error": f"Failed to fetch news for {ticker}"}]

# --- Main Execution ---
if __name__ == "__main__":
    log.info("Starting Finance Data MCP Server (v2)...")
    log.info(f"Finnhub Key Loaded: {'Yes' if settings.finnhub_api_key else 'No'}")
    log.info(f"AlphaVantage Key Loaded: {'Yes' if settings.alpha_vantage_api_key else 'No'}")
    if not settings.finnhub_api_key and not settings.alpha_vantage_api_key:
        log.warning("\n*** WARNING: No API keys found. Most features will not work. ***")
        log.warning("*** Create a .env file with FINNHUB_API_KEY and ALPHA_VANTAGE_API_KEY ***\n")
    elif not settings.alpha_vantage_api_key:
         log.warning("\n*** WARNING: ALPHA_VANTAGE_API_KEY not found. Market movers feature disabled. ***\n")
    elif not settings.finnhub_api_key:
         log.warning("\n*** WARNING: FINNHUB_API_KEY not found. News features disabled. ***\n")

    async def close_http_client():
        if not http_client.is_closed:
            await http_client.aclose()
            log.info("HTTP client closed.")

    async def main_run():
        try:
            await mcp.run_async()
        finally:
            await close_http_client()

    try:
        asyncio.run(main_run())
    except KeyboardInterrupt:
         log.info("Server stopped by user.")
    finally:
         # Ensure client is closed even on KeyboardInterrupt during startup
         if not http_client.is_closed:
              try:
                   asyncio.run(close_http_client())
              except RuntimeError: # Event loop might already be closed
                   pass