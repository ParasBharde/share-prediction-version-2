"""
SmartAPI Option-Chain Relay (local helper)

Purpose:
    Run a local HTTP relay that authenticates with Angel One SmartAPI and
    exposes a scanner-friendly endpoint:

        GET /option-chain/{symbol}

    The relay fetches from a user-configured SmartAPI option-chain endpoint,
    normalizes the payload to NSE-like raw schema (`records` root), and returns
    JSON compatible with OptionChainFetcher parsing.

Why this script exists:
    Some networks/IPs are blocked by NSE Akamai for direct option-chain calls.
    This relay lets you use your broker-side API credentials instead.

Required env vars:
    SMARTAPI_API_KEY
    SMARTAPI_CLIENT_CODE
    SMARTAPI_PIN
    SMARTAPI_TOTP_SECRET
    SMARTAPI_OPTION_CHAIN_URL

Optional env vars:
    SMARTAPI_BIND_HOST=127.0.0.1
    SMARTAPI_BIND_PORT=8787

SMARTAPI_OPTION_CHAIN_URL formats supported:
    - https://api.example.com/option-chain/{symbol}
    - https://api.example.com/option-chain?symbol=NIFTY

Usage:
    python scripts/smartapi_option_chain_relay.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pyotp
from aiohttp import ClientSession, web
from SmartApi import SmartConnect


@dataclass
class SmartAPISession:
    api_key: str
    client_code: str
    pin: str
    totp_secret: str
    option_chain_url: str
    jwt_token: str = ""
    refresh_token: str = ""

    def login(self) -> None:
        smart = SmartConnect(api_key=self.api_key)
        otp = pyotp.TOTP(self.totp_secret).now()
        resp = smart.generateSession(self.client_code, self.pin, otp)
        data = (resp or {}).get("data", {})
        self.jwt_token = data.get("jwtToken", "")
        self.refresh_token = data.get("refreshToken", "")
        if not self.jwt_token:
            raise RuntimeError(f"SmartAPI login failed: {resp}")


def _build_url(template: str, symbol: str) -> str:
    if "{symbol}" in template:
        return template.format(symbol=symbol)
    if "symbol=" in template:
        return template
    sep = "&" if "?" in template else "?"
    return f"{template}{sep}symbol={symbol}"


def _normalize_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict) and payload.get("records"):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "result", "payload"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and candidate.get("records"):
                return candidate
    return None


async def _fetch_option_chain(
    http: ClientSession,
    auth: SmartAPISession,
    symbol: str,
) -> Dict[str, Any]:
    if not auth.jwt_token:
        auth.login()

    url = _build_url(auth.option_chain_url, symbol)
    headers = {
        "Authorization": f"Bearer {auth.jwt_token}",
        "Accept": "application/json, text/plain, */*",
    }

    async with http.get(url, headers=headers) as resp:
        body = await resp.json(content_type=None)
        normalized = _normalize_payload(body)
        if not normalized:
            raise web.HTTPBadGateway(
                text=(
                    "Upstream SmartAPI payload is not NSE-like. "
                    "Expected records root or wrapped records under "
                    "data/result/payload."
                )
            )
        return normalized


async def _create_app() -> web.Application:
    cfg = SmartAPISession(
        api_key=os.environ["SMARTAPI_API_KEY"],
        client_code=os.environ["SMARTAPI_CLIENT_CODE"],
        pin=os.environ["SMARTAPI_PIN"],
        totp_secret=os.environ["SMARTAPI_TOTP_SECRET"],
        option_chain_url=os.environ["SMARTAPI_OPTION_CHAIN_URL"],
    )

    app = web.Application()
    app["smartapi_auth"] = cfg
    app["http"] = ClientSession()

    async def option_chain_handler(request: web.Request) -> web.Response:
        symbol = request.match_info.get("symbol", "NIFTY").upper()
        payload = await _fetch_option_chain(request.app["http"], request.app["smartapi_auth"], symbol)
        return web.json_response(payload)

    async def health_handler(_: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def on_cleanup(app_: web.Application) -> None:
        await app_["http"].close()

    app.router.add_get("/health", health_handler)
    app.router.add_get("/option-chain/{symbol}", option_chain_handler)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    host = os.environ.get("SMARTAPI_BIND_HOST", "127.0.0.1")
    port = int(os.environ.get("SMARTAPI_BIND_PORT", "8787"))

    app = asyncio.run(_create_app())
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
