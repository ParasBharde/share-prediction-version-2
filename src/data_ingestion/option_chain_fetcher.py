"""
Option Chain Data Fetcher

Fetches option chain data from NSE for NIFTY, BANKNIFTY, and individual stocks.
Provides OI, IV, premium, Greeks data for options strategies.

Fetch strategy (in priority order):
    1. nsepython.nse_optionchain_scrapper() — battle-tested NSE library that
       manages sessions/cookies internally. Runs in a thread via asyncio.to_thread.
    2. Playwright (real Chromium browser) — executes JavaScript so Akamai's
       _abck cookie validation passes. The only reliable way to get live NSE
       data without an official API key.
       Install with: pip install playwright && playwright install chromium
    3. aiohttp browser-simulation fallback — last resort; always blocked by
       Akamai because it cannot execute JavaScript to validate _abck.

NSE Akamai Anti-Bot Notes:
    NSE uses Akamai Bot Manager. The _abck cookie requires JavaScript execution
    to validate. Pure Python HTTP clients (requests, aiohttp, nsepython,
    curl_cffi) all receive a placeholder _abck cookie. NSE's server silently
    returns HTTP 200 with body {} for any request whose _abck is invalid.
    Playwright runs real Chrome JS which generates a valid _abck cookie.
"""

import asyncio
import random
import time
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from src.monitoring.logger import get_logger

# ── nsepython availability check ────────────────────────────────────────────
try:
    from nsepython import nse_optionchain_scrapper as _nse_scrapper
    _NSEPYTHON_OK = True
except ImportError:  # pragma: no cover
    _NSEPYTHON_OK = False
    _nse_scrapper = None  # type: ignore[assignment]

# ── Playwright availability check (real browser — bypasses Akamai fully) ────
# Install with: pip install playwright && playwright install chromium
try:
    from playwright.async_api import async_playwright as _async_playwright  # type: ignore[import]
    _PLAYWRIGHT_OK = True
except ImportError:
    _PLAYWRIGHT_OK = False
    _async_playwright = None  # type: ignore[assignment]

logger = get_logger(__name__)

# ── NSE API endpoints ────────────────────────────────────────────────────────
NSE_BASE_URL        = "https://www.nseindia.com"
NSE_OPTION_CHAIN_URL = "https://www.nseindia.com/api/option-chain-indices"
NSE_EQUITY_OPTION_URL = "https://www.nseindia.com/api/option-chain-equities"

# ── Browser User-Agent rotation pool ────────────────────────────────────────
# Use realistic, recent browser strings. Rotate to avoid fingerprinting.
_USER_AGENTS = [
    # Chrome 124 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    # Chrome 123 on macOS
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    # Firefox 125 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
    # Edge 124 on Windows
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
    ),
    # Chrome 122 on Linux
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
]

# ── Warmup page sequence – mimics a real trader browsing NSE ─────────────────
_WARMUP_PAGES = [
    "/",
    "/market-data/live-equity-market",
    "/option-chain",
]

# ── Indices that have weekly options (use nearest expiry logic) ───────────────
_WEEKLY_EXPIRY_INDICES = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX"}

# Maximum consecutive auth failures before full session recreation
_MAX_AUTH_FAILURES = 3


def _build_headers(user_agent: str, referer: str = NSE_BASE_URL) -> Dict[str, str]:
    """Build a realistic browser-like header set for NSE requests."""
    return {
        "User-Agent": user_agent,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,image/apng,*/*;q=0.8,"
            "application/signed-exchange;v=b3;q=0.7"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
        "Referer": referer,
        "DNT": "1",
    }


def _build_api_headers(user_agent: str, referer: str) -> Dict[str, str]:
    """Build headers for NSE JSON API calls (XHR-style)."""
    return {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": referer,
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "X-Requested-With": "XMLHttpRequest",
        "DNT": "1",
    }


def _is_expiry_day(expiry_date_str: str) -> bool:
    """
    Check whether a given NSE expiry date string is today.

    NSE expiry dates come as 'DD-Mon-YYYY' (e.g., '25-Apr-2024').
    """
    try:
        expiry = datetime.strptime(expiry_date_str, "%d-%b-%Y").date()
        return expiry == date.today()
    except ValueError:
        return False


def _top_oi_clusters(
    strikes: List[Dict],
    side: str,          # "ce_oi" or "pe_oi"
    top_n: int = 3,
) -> List[Dict]:
    """
    Return the top-N strikes by OI for more reliable support/resistance.

    Using only the single max OI strike is fragile — OI can shift intraday.
    Returning top-3 clusters gives a band of key levels.
    """
    sorted_strikes = sorted(
        [s for s in strikes if s.get(side, 0) > 0],
        key=lambda x: x.get(side, 0),
        reverse=True,
    )
    return sorted_strikes[:top_n]


class OptionChainFetcher:
    """
    Fetches option chain data from NSE India with robust anti-block logic.

    The class maintains a long-lived aiohttp session with proper NSE cookies.
    Sessions are warmed up by visiting browser-like page sequences before
    calling the JSON API, which prevents the most common 401/403 responses.
    """

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None
        self._user_agent: str = random.choice(_USER_AGENTS)
        self._auth_failures: int = 0
        self._last_warmup: Optional[datetime] = None
        # Re-warm session every 4 minutes to keep cookies fresh
        self._warmup_ttl_seconds: int = 240
        # Playwright browser (lazily initialised on first use)
        self._pw_playwright = None
        self._pw_browser = None

    # ── Session management ────────────────────────────────────────────────────

    async def _create_session(self) -> aiohttp.ClientSession:
        """Create a brand-new aiohttp session with a fresh cookie jar."""
        timeout = aiohttp.ClientTimeout(total=45, connect=15, sock_read=20)
        session = aiohttp.ClientSession(
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar(),
            connector=aiohttp.TCPConnector(ssl=True, limit=5),
        )
        return session

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get the current session, creating one if necessary."""
        if self._session is None or self._session.closed:
            self._session = await self._create_session()
            self._last_warmup = None   # force warmup on new session
        return self._session

    async def _ensure_warmed_up(self) -> bool:
        """
        Ensure the session has valid NSE cookies by running the warmup
        sequence if cookies are stale or absent.

        Returns:
            True if warmup succeeded (cookies obtained), False otherwise.
        """
        now = datetime.now()
        if self._last_warmup is not None:
            age = (now - self._last_warmup).total_seconds()
            if age < self._warmup_ttl_seconds:
                return True   # cookies still fresh

        # Rotate user-agent on each warmup cycle
        self._user_agent = random.choice(_USER_AGENTS)
        session = await self._get_session()

        for page in _WARMUP_PAGES:
            url = f"{NSE_BASE_URL}{page}"
            try:
                headers = _build_headers(
                    self._user_agent,
                    referer=NSE_BASE_URL if page == "/" else f"{NSE_BASE_URL}/",
                )
                async with session.get(
                    url,
                    headers=headers,
                    allow_redirects=True,
                ) as resp:
                    await resp.read()
                    status = resp.status
                    logger.debug(
                        f"NSE warmup {page} → HTTP {status}"
                    )
                    if status in (429, 503):
                        # Rate limited — wait longer
                        retry_after = int(
                            resp.headers.get("Retry-After", "10")
                        )
                        logger.warning(
                            f"NSE warmup rate-limited on {page}, "
                            f"sleeping {retry_after}s"
                        )
                        await asyncio.sleep(retry_after + random.uniform(1, 3))
            except Exception as exc:
                logger.debug(f"NSE warmup failed on {page}: {exc}")

            # Human-like delay between page visits
            await asyncio.sleep(random.uniform(0.8, 2.0))

        # Check if we obtained the critical cookies
        cookies = {c.key: c.value for c in session.cookie_jar}
        has_cookies = bool(cookies)
        if has_cookies:
            self._last_warmup = now
            self._auth_failures = 0
            logger.info(
                f"NSE session warmed up successfully "
                f"(cookies: {list(cookies.keys())})"
            )
        else:
            logger.warning("NSE warmup completed but no cookies received")

        return has_cookies

    async def _recreate_session(self) -> None:
        """Destroy and recreate the session (used after repeated auth failures)."""
        logger.warning(
            "Recreating NSE session after repeated auth failures"
        )
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.25)
        self._session = None
        self._last_warmup = None
        self._auth_failures = 0
        # Longer sleep before retry to let any IP-level block expire
        await asyncio.sleep(random.uniform(5, 10))

    # ── nsepython primary fetch ───────────────────────────────────────────────

    async def _fetch_via_nsepython(self, symbol: str) -> Optional[Dict]:
        """
        Fetch raw NSE option chain JSON via nsepython (synchronous, run in thread).

        nsepython.nse_optionchain_scrapper() manages its own requests.Session,
        correctly negotiates NSE cookies (including Akamai), and returns the
        same JSON structure that NSE's API returns.

        Returns the raw dict (same schema accepted by _parse_option_chain),
        or None if nsepython is not installed or the call fails.
        """
        if not _NSEPYTHON_OK or _nse_scrapper is None:
            logger.debug("nsepython not available — skipping primary fetch path")
            return None

        try:
            logger.info(f"Fetching {symbol} option chain via nsepython …")
            # Run synchronous nsepython in a thread so we don't block the event loop
            raw = await asyncio.to_thread(_nse_scrapper, symbol)

            if raw and isinstance(raw, dict) and raw.get("records"):
                logger.info(
                    f"nsepython fetch OK for {symbol} "
                    f"({len(raw.get('records', {}).get('data', []))} records)"
                )
                return raw

            # ── Detailed diagnostics so we can see EXACTLY what came back ─────
            if isinstance(raw, dict):
                top_keys = list(raw.keys())
                preview   = str(raw)[:400]
                logger.warning(
                    f"nsepython returned a dict WITHOUT 'records' for {symbol}. "
                    f"Top-level keys: {top_keys}. "
                    f"Content preview: {preview!r}"
                )
                # ── Alternate structure: some NSE API versions nest differently ─
                # If 'filtered' exists but 'records' is missing, try wrapping it
                if raw.get("filtered") and not raw.get("records"):
                    logger.info(
                        f"Attempting to reconstruct 'records' from 'filtered' "
                        f"for {symbol} …"
                    )
                    # Build a minimal records block from filtered so _parse_option_chain
                    # can still run — underlying price may be 0 but OI data will work
                    synthetic = {
                        "records": {
                            "data":           raw["filtered"].get("data", []),
                            "expiryDates":    raw.get("expiryDates", []),
                            "underlyingValue": raw.get("underlyingValue", 0.0),
                        },
                        "filtered": raw["filtered"],
                    }
                    if synthetic["records"]["data"]:
                        logger.info(
                            f"Reconstructed option chain for {symbol} from "
                            f"'filtered' data "
                            f"({len(synthetic['records']['data'])} rows)"
                        )
                        return synthetic
            else:
                logger.warning(
                    f"nsepython returned non-dict for {symbol}: "
                    f"type={type(raw).__name__}, value={str(raw)[:200]!r}"
                )
        except Exception as exc:
            logger.warning(f"nsepython fetch failed for {symbol}: {exc}")
        return None

    # ── Playwright real-browser fetch ─────────────────────────────────────────

    async def _ensure_playwright(self) -> None:
        """Launch the Playwright Chromium browser if not already running."""
        if self._pw_browser is not None:
            try:
                if self._pw_browser.is_connected():
                    return
            except Exception:
                pass
            # Browser disconnected — reset
            self._pw_browser = None
            self._pw_playwright = None

        if not _PLAYWRIGHT_OK or _async_playwright is None:
            raise RuntimeError("playwright not installed — run: pip install playwright && playwright install chromium")

        self._pw_playwright = await _async_playwright().__aenter__()
        self._pw_browser = await self._pw_playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        logger.info("Playwright Chromium browser launched")

    async def _fetch_via_playwright(self, symbol: str) -> Optional[Dict]:
        """
        Fetch NSE option chain using a real Chromium browser via Playwright.

        Playwright executes JavaScript, which causes Akamai's _abck cookie
        validation to succeed — the only reliable way to bypass NSE's bot
        detection without an official API key.

        The browser is kept alive between calls for performance; only the
        first call incurs the browser-launch overhead (~2–5 s).

        Install with: pip install playwright && playwright install chromium
        """
        if not _PLAYWRIGHT_OK:
            return None

        try:
            await self._ensure_playwright()
        except Exception as exc:
            logger.warning(f"Playwright init failed: {exc}")
            return None

        if symbol in _WEEKLY_EXPIRY_INDICES:
            api_path_fragment = "/api/option-chain-indices"
        else:
            api_path_fragment = "/api/option-chain-equities"

        try:
            logger.info(
                f"Fetching {symbol} option chain via Playwright (real browser) …"
            )
            context = await self._pw_browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
            )
            page = await context.new_page()

            # Intercept the XHR response that NSE's JS makes to the API
            captured: Dict = {}

            async def _on_response(response) -> None:
                if api_path_fragment in response.url and response.status == 200:
                    try:
                        data = await response.json()
                        if isinstance(data, dict) and data.get("records"):
                            captured["data"] = data
                    except Exception:
                        pass

            page.on("response", _on_response)

            try:
                await page.goto(
                    f"{NSE_BASE_URL}/option-chain?symbol={symbol}",
                    wait_until="domcontentloaded",
                    timeout=30_000,
                )
                # Wait for NSE's JavaScript to fire the option chain XHR
                await asyncio.sleep(4)

                raw = captured.get("data")
                if not raw:
                    # Fallback: call API directly from the browser's JS context
                    # (cookies already set by Akamai at this point)
                    raw = await page.evaluate(
                        f"""async () => {{
                            try {{
                                const r = await fetch(
                                    '{api_path_fragment}?symbol={symbol}',
                                    {{credentials: 'include',
                                     headers: {{'Accept': 'application/json'}}}}
                                );
                                return r.ok ? await r.json() : null;
                            }} catch(e) {{ return null; }}
                        }}"""
                    )

                if raw and isinstance(raw, dict) and raw.get("records"):
                    logger.info(
                        f"Playwright fetch OK for {symbol} "
                        f"({len(raw.get('records', {}).get('data', []))} records)"
                    )
                    return raw

                logger.warning(
                    f"Playwright fetch yielded no valid data for {symbol}. "
                    f"keys={list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
                )
            finally:
                await page.close()
                await context.close()

        except Exception as exc:
            logger.warning(f"Playwright fetch failed for {symbol}: {exc}")
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_option_chain(
        self,
        symbol: str = "NIFTY",
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch full option chain data for a symbol.

        Fetch order:
          1. nsepython.nse_optionchain_scrapper() — handles Akamai cookies natively
          2. aiohttp browser-simulation fallback (5 attempts with backoff)

        Args:
            symbol: Index name (NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY)
                    or equity symbol.

        Returns:
            Parsed option chain dict or None on complete failure.
        """
        symbol = symbol.upper()

        # ── 1. Try nsepython first ────────────────────────────────────────────
        raw = await self._fetch_via_nsepython(symbol)
        if raw:
            return self._parse_option_chain(raw, symbol)

        # ── 2. Playwright real browser (executes JS — fully bypasses Akamai) ───
        if _PLAYWRIGHT_OK:
            raw = await self._fetch_via_playwright(symbol)
            if raw:
                return self._parse_option_chain(raw, symbol)
        else:
            logger.warning(
                "playwright not installed — Akamai will block all pure-Python "
                "HTTP clients. Fix: pip install playwright && playwright install chromium"
            )

        # ── 3. Fallback: aiohttp browser-simulation (blocked by Akamai) ───────
        logger.info(
            f"nsepython/playwright failed for {symbol} — "
            f"falling back to aiohttp (likely blocked by Akamai)"
        )

        # Choose the correct endpoint
        if symbol in _WEEKLY_EXPIRY_INDICES:
            url = NSE_OPTION_CHAIN_URL
        else:
            url = NSE_EQUITY_OPTION_URL

        params = {"symbol": symbol}
        symbol_page_url = f"{NSE_BASE_URL}/option-chain?symbol={symbol}"
        api_referer = symbol_page_url

        max_attempts = 5
        base_delay = 2.0   # seconds

        for attempt in range(max_attempts):
            # Ensure cookies are warm before the API call
            warmed = await self._ensure_warmed_up()
            if not warmed:
                logger.warning(
                    f"Warmup failed on attempt {attempt + 1}/{max_attempts}"
                )

            # Visit the symbol-specific option chain page before the API call
            session = await self._get_session()
            try:
                async with session.get(
                    symbol_page_url,
                    headers=_build_headers(self._user_agent, referer=f"{NSE_BASE_URL}/"),
                    allow_redirects=True,
                ) as page_resp:
                    await page_resp.read()
                    logger.debug(
                        f"NSE symbol page {symbol} → HTTP {page_resp.status}"
                    )
            except Exception as page_exc:
                logger.debug(
                    f"NSE symbol page visit failed for {symbol}: {page_exc}"
                )
            await asyncio.sleep(random.uniform(1.0, 2.0))

            headers = _build_api_headers(self._user_agent, api_referer)

            try:
                async with session.get(
                    url,
                    headers=headers,
                    params=params,
                ) as resp:
                    status = resp.status
                    logger.debug(
                        f"NSE option chain {symbol} "
                        f"attempt {attempt + 1} → HTTP {status}"
                    )

                    if status == 200:
                        try:
                            data = await resp.json(content_type=None)
                        except Exception as parse_exc:
                            logger.warning(
                                f"JSON parse error for {symbol}: {parse_exc}"
                            )
                            data = None

                        if data:
                            self._auth_failures = 0
                            return self._parse_option_chain(data, symbol)
                        logger.warning(
                            f"NSE returned empty body for {symbol} "
                            f"(attempt {attempt + 1}) — "
                            f"Akamai bot-detection triggered. "
                            f"Fix: pip install curl_cffi"
                        )

                    elif status in (401, 403):
                        self._auth_failures += 1
                        logger.warning(
                            f"NSE auth failure ({status}) for {symbol} "
                            f"(consecutive: {self._auth_failures})"
                        )
                        if self._auth_failures >= _MAX_AUTH_FAILURES:
                            await self._recreate_session()
                        else:
                            self._last_warmup = None

                    elif status == 429:
                        retry_after = int(
                            resp.headers.get("Retry-After", str(int(base_delay * 4)))
                        )
                        logger.warning(
                            f"NSE rate-limited (429) for {symbol}, "
                            f"sleeping {retry_after}s"
                        )
                        await asyncio.sleep(
                            retry_after + random.uniform(1, 3)
                        )
                        continue

                    elif status >= 500:
                        logger.warning(
                            f"NSE server error ({status}) for {symbol}"
                        )

                    else:
                        logger.warning(
                            f"Unexpected HTTP {status} for {symbol} "
                            f"option chain"
                        )

            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout fetching {symbol} option chain "
                    f"(attempt {attempt + 1})"
                )
            except aiohttp.ClientConnectorError as exc:
                logger.warning(
                    f"Connection error for {symbol}: {exc}"
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except Exception as exc:
                logger.warning(
                    f"Unexpected error fetching {symbol} "
                    f"option chain: {exc}"
                )

            # Exponential backoff with jitter before next attempt
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                delay = min(delay, 30)   # cap at 30 seconds
                logger.debug(
                    f"Retrying {symbol} option chain in {delay:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)

        logger.error(
            f"All {max_attempts} aiohttp attempts failed for {symbol} option chain. "
            f"Akamai is blocking pure-Python HTTP clients. "
            f"Install curl_cffi for Chrome TLS impersonation: pip install curl_cffi"
        )
        return None

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse_option_chain(
        self,
        raw_data: Dict,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Parse NSE option chain JSON into a structured dict.

        Improvements over the original:
        - Expiry-day awareness: uses next expiry if today is expiry day
        - Top-3 OI clustering for robust support/resistance
        - ATM strike calculation included
        - IV data preserved per strike
        """
        records  = raw_data.get("records", {})
        filtered = raw_data.get("filtered", {})

        # Current underlying spot price
        underlying_price = records.get("underlyingValue", 0.0)

        # ── Expiry selection ──────────────────────────────────────────────────
        expiry_dates = records.get("expiryDates", [])
        selected_expiry = ""
        selected_expiry_idx = 0

        if expiry_dates:
            # On expiry day itself, next expiry has more relevant OI
            if _is_expiry_day(expiry_dates[0]):
                logger.info(
                    f"Today is expiry day for {symbol}. "
                    f"Switching to next expiry: "
                    f"{expiry_dates[1] if len(expiry_dates) > 1 else 'N/A'}"
                )
                selected_expiry_idx = 1 if len(expiry_dates) > 1 else 0
            selected_expiry = expiry_dates[selected_expiry_idx]

        is_expiry_day_flag = (
            len(expiry_dates) > 0 and _is_expiry_day(expiry_dates[0])
        )

        # ── Strike parsing ────────────────────────────────────────────────────
        # Use 'filtered' data (near ATM) when available — it's cleaner
        all_data = filtered.get("data", []) or records.get("data", [])

        strikes: List[Dict] = []
        total_ce_oi = 0
        total_pe_oi = 0

        for row in all_data:
            strike_price = row.get("strikePrice", 0)
            ce = row.get("CE", {}) or {}
            pe = row.get("PE", {}) or {}

            ce_oi         = ce.get("openInterest",          0) or 0
            pe_oi         = pe.get("openInterest",          0) or 0
            ce_oi_change  = ce.get("changeinOpenInterest",  0) or 0
            pe_oi_change  = pe.get("changeinOpenInterest",  0) or 0
            ce_iv         = ce.get("impliedVolatility",     0) or 0
            pe_iv         = pe.get("impliedVolatility",     0) or 0
            ce_ltp        = ce.get("lastPrice",             0) or 0
            pe_ltp        = pe.get("lastPrice",             0) or 0
            ce_volume     = ce.get("totalTradedVolume",     0) or 0
            pe_volume     = pe.get("totalTradedVolume",     0) or 0
            ce_bid        = ce.get("bidprice",              0) or 0
            ce_ask        = ce.get("askPrice",              0) or 0
            pe_bid        = pe.get("bidprice",              0) or 0
            pe_ask        = pe.get("askPrice",              0) or 0

            total_ce_oi += ce_oi
            total_pe_oi += pe_oi

            strikes.append({
                "strike":       strike_price,
                "ce_oi":        ce_oi,
                "pe_oi":        pe_oi,
                "ce_oi_change": ce_oi_change,
                "pe_oi_change": pe_oi_change,
                "ce_iv":        ce_iv,
                "pe_iv":        pe_iv,
                "ce_ltp":       ce_ltp,
                "pe_ltp":       pe_ltp,
                "ce_volume":    ce_volume,
                "pe_volume":    pe_volume,
                "ce_bid":       ce_bid,
                "ce_ask":       ce_ask,
                "pe_bid":       pe_bid,
                "pe_ask":       pe_ask,
            })

        # ── PCR ──────────────────────────────────────────────────────────────
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0.0

        # ── Top-3 OI clusters for support / resistance ────────────────────────
        top_ce_clusters = _top_oi_clusters(strikes, "ce_oi", top_n=3)
        top_pe_clusters = _top_oi_clusters(strikes, "pe_oi", top_n=3)

        # Primary support/resistance = single highest OI strike
        max_ce_oi_strike = top_ce_clusters[0]["strike"] if top_ce_clusters else 0
        max_pe_oi_strike = top_pe_clusters[0]["strike"] if top_pe_clusters else 0

        # ── ATM strike and option premiums ────────────────────────────────────
        atm_strike = _find_atm_strike(strikes, underlying_price)
        atm_ce_ltp, atm_pe_ltp, atm_ce_iv, atm_pe_iv = 0.0, 0.0, 0.0, 0.0
        for s in strikes:
            if s["strike"] == atm_strike:
                atm_ce_ltp = s["ce_ltp"]
                atm_pe_ltp = s["pe_ltp"]
                atm_ce_iv  = s["ce_iv"]
                atm_pe_iv  = s["pe_iv"]
                break

        return {
            "symbol":           symbol,
            "underlying_price": underlying_price,
            "current_expiry":   selected_expiry,
            "expiry_dates":     expiry_dates,
            "is_expiry_day":    is_expiry_day_flag,
            "total_ce_oi":      total_ce_oi,
            "total_pe_oi":      total_pe_oi,
            "pcr":              round(pcr, 3),
            # Primary levels (single max OI – backwards compatible)
            "max_ce_oi_strike": max_ce_oi_strike,
            "max_ce_oi":        top_ce_clusters[0]["ce_oi"] if top_ce_clusters else 0,
            "max_pe_oi_strike": max_pe_oi_strike,
            "max_pe_oi":        top_pe_clusters[0]["pe_oi"] if top_pe_clusters else 0,
            "support":          max_pe_oi_strike,    # Max PUT OI = support
            "resistance":       max_ce_oi_strike,    # Max CALL OI = resistance
            # Top-3 OI clusters for band-based analysis
            "ce_resistance_levels": [c["strike"] for c in top_ce_clusters],
            "pe_support_levels":    [c["strike"] for c in top_pe_clusters],
            # ATM option premiums (critical for actual trading)
            "atm_strike":   atm_strike,
            "atm_ce_ltp":   atm_ce_ltp,
            "atm_pe_ltp":   atm_pe_ltp,
            "atm_ce_iv":    atm_ce_iv,
            "atm_pe_iv":    atm_pe_iv,
            # All strikes (for strategy-level analysis)
            "strikes": strikes,
        }

    async def close(self) -> None:
        """Close the HTTP session and Playwright browser (if open)."""
        if self._pw_browser is not None:
            try:
                await self._pw_browser.close()
            except Exception:
                pass
            self._pw_browser = None
        if self._pw_playwright is not None:
            try:
                await self._pw_playwright.__aexit__(None, None, None)
            except Exception:
                pass
            self._pw_playwright = None
        if self._session and not self._session.closed:
            await self._session.close()
            await asyncio.sleep(0.25)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_atm_strike(strikes: List[Dict], spot: float) -> float:
    """Return the strike closest to the current spot price."""
    if not strikes or spot <= 0:
        return 0.0
    return min(strikes, key=lambda s: abs(s["strike"] - spot))["strike"]
