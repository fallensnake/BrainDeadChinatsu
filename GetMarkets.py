#!/usr/bin/env python3
"""
Step 1: Pull historical NBA market data from Kalshi for the 2024-25 and 2025-26 seasons.

For every NBA market in the date range this collects:
  1. price_at_start_*  -> the contract price at (approximately) game tip-off, from 1-minute
     candlesticks. Old-script difference: your previous code read the LIVE snapshot (price at
     pull time). Here we pull historical candlesticks and look up the candle covering tip-off.
  2. result_binary     -> 1 if the market settled YES, 0 if NO, None if void/unsettled.
  3. metadata          -> market title, event_ticker (for grouping), series_ticker, and an
     inferred market_type (game_winner / spread / total / series / champion / player_prop / ...).

Output: one row per market in outputs/nba_kalshi_markets.csv (resumable). Optionally dumps the
raw tip-off candle window per market for re-anchoring later.

NO AUTH is required for Kalshi market-data + candlestick endpoints. If you hit 401s, set
KALSHI_KEY_ID + KALSHI_PRIVATE_KEY_PATH env vars and the script will sign requests (RSA-PSS).

  pip install requests pandas
  pip install cryptography   # only if you need authenticated requests
  python pull_nba_kalshi.py
"""

from __future__ import annotations
import os, sys, csv, time, json, base64, datetime as dt
from typing import Optional, Iterable
import requests

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
BASE = "https://api.elections.kalshi.com/trade-api/v2"

# Season windows (UTC). NBA season ~ Oct -> June. Widened a little to be safe.
SEASON_START = dt.datetime(2024, 9, 1, tzinfo=dt.timezone.utc)
SEASON_END   = dt.datetime(2026, 7, 15, tzinfo=dt.timezone.utc)

# Tip-off anchor: how we estimate the candle to read the "price at event start".
# We pull 1-min candles in [anchor - PRE_MIN, anchor + POST_MIN] and take the last candle
# at or before the anchor (i.e. the final price before tip-off).
PRE_MIN  = 90
POST_MIN = 20
CANDLE_PERIOD_MIN = 1            # 1-minute candles (valid: 1, 60, 1440)

# Fallback if no event.strike_date: expected_expiration is ~ a few hrs AFTER start, so we
# subtract this to approximate tip-off. NBA games run ~2.5h. Only used as a fallback.
EXPECTED_TO_TIPOFF_MIN = 150

OUT_CSV     = "nba_kalshi_markets.csv"
RAW_CANDLES = "nba_kalshi_tipoff_candles.jsonl"   # set to None to skip raw dump
SAVE_RAW    = True

SLEEP = 0.12                      # polite pause between calls; raise if you see 429s
MAX_RETRIES = 5

# Manually force-include / exclude series tickers here if auto-discovery misses something.
SERIES_INCLUDE_OVERRIDE: list[str] = []   # e.g. ["KXNBAGAME", "KXNBASERIES"]
SERIES_EXCLUDE = {"WNBA"}                  # substrings to drop (WNBA != NBA)

# --------------------------------------------------------------------------------------
# AUTH (optional) — only used if KALSHI_KEY_ID is set
# --------------------------------------------------------------------------------------
KEY_ID = os.environ.get("KALSHI_KEY_ID")
PRIV_PATH = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
_private_key = None
if KEY_ID and PRIV_PATH:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    with open(PRIV_PATH, "rb") as f:
        _private_key = serialization.load_pem_private_key(f.read(), password=None)

def _auth_headers(method: str, path: str) -> dict:
    if not _private_key:
        return {}
    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + path).encode()
    sig = _private_key.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }

# --------------------------------------------------------------------------------------
# HTTP helper
# --------------------------------------------------------------------------------------
SESSION = requests.Session()

def get(path: str, params: dict | None = None) -> Optional[dict]:
    """GET BASE+path. Returns parsed JSON, or None on 404/empty. Retries on 429/5xx."""
    url = BASE + path
    # path used for signing must include the query string; build it the same way requests will
    sign_path = "/trade-api/v2" + path
    for attempt in range(MAX_RETRIES):
        headers = _auth_headers("GET", sign_path)
        try:
            r = SESSION.get(url, params=params, headers=headers, timeout=30)
        except requests.RequestException as e:
            time.sleep(1.5 * (attempt + 1))
            continue
        if r.status_code == 200:
            time.sleep(SLEEP)
            return r.json()
        if r.status_code == 404:
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            wait = 2 ** attempt
            print(f"  {r.status_code} on {path} -> retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code in (401, 403):
            print(f"  AUTH ERROR {r.status_code} on {path}. Set KALSHI_KEY_ID + "
                  f"KALSHI_PRIVATE_KEY_PATH to sign requests.", file=sys.stderr)
            return None
        print(f"  HTTP {r.status_code} on {path}: {r.text[:200]}", file=sys.stderr)
        return None
    return None

def paginate(path: str, params: dict, key: str) -> Iterable[dict]:
    """Yield items from a cursor-paginated list endpoint."""
    params = dict(params)
    while True:
        data = get(path, params)
        if not data:
            return
        for item in data.get(key, []) or []:
            yield item
        cursor = data.get("cursor")
        if not cursor:
            return
        params["cursor"] = cursor

# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------
def to_ts(iso: str | None) -> Optional[int]:
    if not iso:
        return None
    try:
        return int(dt.datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None

def price_to_prob(v) -> Optional[float]:
    """Normalize a Kalshi price to a 0..1 probability. Handles dollar strings ('0.56'),
    cent ints (56), and None."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f <= 1.0 else f / 100.0

def candle_field(candle: dict, field: str) -> Optional[float]:
    """Pull a close price from a candle, tolerating both schemas:
       {'yes_bid': {'close': '0.56'}}  and  {'yes_bid': {'close_dollars': '0.56'}}."""
    block = candle.get(field)
    if not isinstance(block, dict):
        return None
    for k in ("close", "close_dollars", "mean", "mean_dollars"):
        if k in block:
            return price_to_prob(block[k])
    return None

# --------------------------------------------------------------------------------------
# NBA series discovery
# --------------------------------------------------------------------------------------
def discover_nba_series() -> list[str]:
    if SERIES_INCLUDE_OVERRIDE:
        return SERIES_INCLUDE_OVERRIDE
    found: set[str] = set()
    # Try the sports category; fall back to no filter if needed.
    for params in ({"category": "Sports"}, {}):
        data = get("/series/", params) or {}
        for s in data.get("series", []) or []:
            tk = (s.get("ticker") or "").upper()
            if "NBA" in tk and not any(x in tk for x in SERIES_EXCLUDE):
                found.add(s["ticker"])
        if found:
            break
    return sorted(found)

def classify(series_ticker: str, title: str = "") -> str:
    t = series_ticker.upper()
    blob = (series_ticker + " " + (title or "")).upper()
    if "SERIES" in t:                      return "series_winner"
    if t in ("KXNBA",) or "TITLE" in blob or "CHAMPION" in blob: return "champion"
    if "EAST" in t:                        return "east_conf_winner"
    if "WEST" in t:                        return "west_conf_winner"
    if "SPREAD" in blob:                   return "spread"
    if "TOTAL" in blob or "OVER" in blob:  return "total_points"
    if any(k in blob for k in ("POINT", "PTS", "REB", "AST", "ASSIST", "REBOUND",
                               "DOUBLE", "TRIPLE", "BLOCK", "STEAL", "THREE", "3PT")):
        return "player_prop"
    if "GAME" in t or "WINNER" in blob or "BEAT" in blob: return "game_winner"
    return "other"

# --------------------------------------------------------------------------------------
# Markets (live + historical tiers)
# --------------------------------------------------------------------------------------
def list_markets_for_series(series_ticker: str) -> Iterable[dict]:
    """Markets settled before Kalshi's historical cutoff only appear under /historical/markets,
    so we sweep both tiers and de-dup by ticker."""
    seen: set[str] = set()
    for base_path in ("/markets", "/historical/markets"):
        params = {"series_ticker": series_ticker, "limit": 1000}
        for m in paginate(base_path, params, "markets"):
            tk = m.get("ticker")
            if tk and tk not in seen:
                seen.add(tk)
                yield m

def get_event(event_ticker: str, cache: dict) -> dict:
    if event_ticker in cache:
        return cache[event_ticker]
    data = get(f"/events/{event_ticker}", {"with_nested_markets": "false"}) or {}
    ev = data.get("event", {}) or {}
    cache[event_ticker] = ev
    return ev

def fetch_candles(series_ticker: str, ticker: str, start_ts: int, end_ts: int) -> list[dict]:
    """Try the live candlestick endpoint, fall back to the historical one."""
    params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": CANDLE_PERIOD_MIN}
    live = get(f"/series/{series_ticker}/markets/{ticker}/candlesticks", params)
    if live and live.get("candlesticks"):
        return live["candlesticks"]
    hist = get(f"/historical/markets/{ticker}/candlesticks", params)
    if hist and hist.get("candlesticks"):
        return hist["candlesticks"]
    return []

# --------------------------------------------------------------------------------------
# Tip-off anchoring
# --------------------------------------------------------------------------------------
def resolve_tipoff(market: dict, event: dict) -> tuple[Optional[int], str]:
    """Best-effort scheduled tip-off timestamp + which field it came from (so you can audit)."""
    sd = to_ts(event.get("strike_date"))
    if sd:
        return sd, "event.strike_date"
    ee = to_ts(market.get("expected_expiration_time"))
    if ee:
        return ee - EXPECTED_TO_TIPOFF_MIN * 60, "expected_expiration-150m"
    ct = to_ts(market.get("close_time"))
    if ct:
        return ct, "close_time"
    return None, "none"

def pick_start_candle(candles: list[dict], anchor: int) -> Optional[dict]:
    """The last candle at/just before tip-off (final price before the game starts)."""
    before = [c for c in candles if c.get("end_period_ts", 0) <= anchor]
    if before:
        return max(before, key=lambda c: c["end_period_ts"])
    return min(candles, key=lambda c: c.get("end_period_ts", 0)) if candles else None

# --------------------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------------------
FIELDS = [
    "market_ticker", "event_ticker", "series_ticker", "market_type",
    "title", "yes_sub_title", "no_sub_title",
    "result", "result_binary",
    "status", "open_time", "close_time", "expected_expiration_time", "strike_date",
    "tipoff_ts_used", "tipoff_source", "candle_ts",
    "price_at_start_yes_bid", "price_at_start_yes_ask", "price_at_start_mid",
    "price_at_start_last", "n_candles_in_window",
    "volume", "open_interest",
]

def load_done(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            done.add(row["market_ticker"])
    return done

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    series_list = discover_nba_series()
    if not series_list:
        print("No NBA series found. Set SERIES_INCLUDE_OVERRIDE manually "
              "(e.g. KXNBAGAME, KXNBASERIES, KXNBA, KXNBAEAST, KXNBAWEST).", file=sys.stderr)
        return
    print(f"NBA series ({len(series_list)}): {', '.join(series_list)}")

    done = load_done(OUT_CSV)
    print(f"Already have {len(done)} markets; resuming.")
    new_file = not os.path.exists(OUT_CSV)
    out = open(OUT_CSV, "a", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    raw = open(RAW_CANDLES, "a") if (SAVE_RAW and RAW_CANDLES) else None

    event_cache: dict = {}
    total = 0
    for series in series_list:
        mtype_default = classify(series)
        print(f"\n=== {series} ({mtype_default}) ===")
        for m in list_markets_for_series(series):
            tk = m.get("ticker")
            if not tk or tk in done:
                continue

            # date-range filter on close_time
            ct = to_ts(m.get("close_time"))
            if ct and not (SEASON_START.timestamp() <= ct <= SEASON_END.timestamp()):
                continue

            ev = get_event(m.get("event_ticker", ""), event_cache)
            anchor, src = resolve_tipoff(m, ev)

            row = {f: None for f in FIELDS}
            row.update({
                "market_ticker": tk,
                "event_ticker": m.get("event_ticker"),
                "series_ticker": series,
                "market_type": classify(series, m.get("title", "")),
                "title": m.get("title"),
                "yes_sub_title": m.get("yes_sub_title"),
                "no_sub_title": m.get("no_sub_title"),
                "result": m.get("result"),
                "result_binary": {"yes": 1, "no": 0}.get((m.get("result") or "").lower()),
                "status": m.get("status"),
                "open_time": m.get("open_time"),
                "close_time": m.get("close_time"),
                "expected_expiration_time": m.get("expected_expiration_time"),
                "strike_date": ev.get("strike_date"),
                "tipoff_ts_used": anchor,
                "tipoff_source": src,
                "volume": m.get("volume_fp") or m.get("volume"),
                "open_interest": m.get("open_interest_fp") or m.get("open_interest"),
            })

            if anchor:
                candles = fetch_candles(series, tk, anchor - PRE_MIN * 60, anchor + POST_MIN * 60)
                row["n_candles_in_window"] = len(candles)
                c = pick_start_candle(candles, anchor)
                if c:
                    yb = candle_field(c, "yes_bid")
                    ya = candle_field(c, "yes_ask")
                    row["price_at_start_yes_bid"] = yb
                    row["price_at_start_yes_ask"] = ya
                    row["price_at_start_mid"] = (yb + ya) / 2 if (yb is not None and ya is not None) else None
                    row["price_at_start_last"] = candle_field(c, "price")
                    row["candle_ts"] = c.get("end_period_ts")
                if raw and candles:
                    raw.write(json.dumps({"ticker": tk, "anchor": anchor, "candles": candles}) + "\n")

            w.writerow(row)
            out.flush()
            total += 1
            if total % 25 == 0:
                print(f"  ...{total} markets written")

    out.close()
    if raw:
        raw.close()
    print(f"\nDone. Wrote {total} new markets to {OUT_CSV}.")

if __name__ == "__main__":
    main()