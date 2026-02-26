# SmartAPI (Angel One) local setup for option-chain scanner

This guide is for users whose direct NSE calls are blocked by Akamai.

## 1) Create SmartAPI app

On `smartapi.angelbroking.com/create`:

- API type: **Trading APIs**
- App name: anything (e.g. `NSEScannerLocal`)
- Redirect URL: `http://127.0.0.1`
- Postback URL: optional (can be blank or `http://127.0.0.1`)
- Angel Client ID: your broker client code

After creating, copy your **API Key**.

## 2) Install dependencies

```bash
pip install smartapi-python pyotp logzero websocket-client pycryptodome
```

## 3) Start local SmartAPI relay

Set environment variables:

```bash
export SMARTAPI_API_KEY="..."
export SMARTAPI_CLIENT_CODE="..."
export SMARTAPI_PIN="..."
export SMARTAPI_TOTP_SECRET="..."

# Endpoint from your SmartAPI option-chain product/docs
# Supported formats:
# - https://host/path/{symbol}
# - https://host/path?symbol=NIFTY
export SMARTAPI_OPTION_CHAIN_URL="https://your-smartapi-endpoint/{symbol}"

python scripts/smartapi_option_chain_relay.py
```

Relay endpoint exposed locally:

- `http://127.0.0.1:8787/option-chain/NIFTY`

## 4) Point scanner to local relay

In a second terminal:

```bash
export OPTION_CHAIN_PROVIDER_URL="http://127.0.0.1:8787/option-chain/{symbol}"
# no token needed for local relay by default
python scripts/options_scan.py --symbol NIFTY --repeat 2
```

## 5) Verify logs

Successful provider fallback logs should include:

- `Fetching NIFTY option chain via configured provider …`
- `Configured provider fetch OK for NIFTY (...)`
