"""
QUANT SCREENER - FastAPI Backend (Finnhub Edition)
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import os
import asyncio
from datetime import datetime
import time
import numpy as np

app = FastAPI(title="Quant Screener API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")
BASE_URL = "https://finnhub.io/api/v1"

US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "TSLA", "AVGO", "AMD", "ORCL",
    "JPM", "V", "MA", "UNH", "LLY",
    "XOM", "CVX", "CAT", "DE", "NEM",
    "FSLR", "COST", "WMT", "HD", "PG",
]

US_SECTOR_ETFS = {
    "Technology": "XLK",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Communication Services": "XLC",
}

BENCHMARK = "SPY"
CACHE_TTL = 1800

_cache: dict = {}


def cache_get(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def cache_set(key, data):
    _cache[key] = (data, time.time())


async def fh_get(client, endpoint, params):
    params["token"] = FINNHUB_KEY
    try:
        r = await client.get(f"{BASE_URL}{endpoint}", params=params, timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


async def get_quote(client, symbol):
    return await fh_get(client, "/quote", {"symbol": symbol})


async def get_metrics(client, symbol):
    return await fh_get(client, "/stock/metric", {"symbol": symbol, "metric": "all"})


async def get_candles(client, symbol, days=180):
    end = int(time.time())
    start = end - (days * 86400)
    return await fh_get(client, "/stock/candle", {
        "symbol": symbol, "resolution": "D", "from": start, "to": end
    })


async def get_profile(client, symbol):
    return await fh_get(client, "/stock/profile2", {"symbol": symbol})


def calc_momentum_score(candles, bm_candles):
    try:
        if not candles or "c" not in candles or not candles["c"]:
            return 50.0
        c = candles["c"]
        if len(c) < 22:
            return 50.0

        now = c[-1]
        m1 = (now / c[-21] - 1) * 100 if len(c) >= 21 else 0
        m3 = (now / c[-63] - 1) * 100 if len(c) >= 63 else 0

        rs = 0.0
        if bm_candles and "c" in bm_candles and len(bm_candles["c"]) >= 21:
            bm = bm_candles["c"]
            bm_ret = (bm[-1] / bm[-21] - 1) * 100
            rs = m1 - bm_ret

        raw = m1 * 0.5 + m3 * 0.3 + rs * 0.2
        return float(np.clip(50 + raw * 1.8, 0, 100))
    except Exception:
        return 50.0


def calc_fundamental_score(metrics):
    try:
        if not metrics or "metric" not in metrics:
            return 50.0
        m = metrics["metric"]
        score = 50.0

        pe = m.get("peNormalizedAnnual") or m.get("peTTM")
        if pe and 0 < pe < 200:
            pe_score = 100 - abs(pe - 20) * 2
            score += (float(np.clip(pe_score, 0, 100)) - 50) * 0.3

        roe = m.get("roeTTM") or m.get("roeRfy")
        if roe:
            roe_score = min(roe * 5, 100)
            score += (roe_score - 50) * 0.3

        eps_growth = m.get("epsGrowth5Y") or m.get("epsGrowthTTMYoy")
        if eps_growth:
            eg_score = min(eps_growth * 3, 100)
            score += (eg_score - 50) * 0.2

        pb = m.get("pbAnnual") or m.get("pbQuarterly")
        if pb and 0 < pb < 50:
            pb_score = 100 - abs(pb - 3) * 5
            score += (float(np.clip(pb_score, 0, 100)) - 50) * 0.2

        return float(np.clip(score, 0, 100))
    except Exception:
        return 50.0


async def analyze_stock(client, symbol, bm_candles, sector_scores):
    candles = await get_candles(client, symbol)
    metrics = await get_metrics(client, symbol)
    profile = await get_profile(client, symbol)

    if not candles or "c" not in candles or not candles["c"]:
        return None

    company = symbol
    sector = "Other"
    if profile:
        company = profile.get("name") or symbol
        sector = profile.get("finnhubIndustry") or "Other"

    pe = None
    roe_str = "-"
    if metrics and "metric" in metrics:
        m = metrics["metric"]
        pe_val = m.get("peNormalizedAnnual") or m.get("peTTM")
        if pe_val and pe_val > 0:
            pe = round(pe_val, 1)
        roe_val = m.get("roeTTM") or m.get("roeRfy")
        if roe_val:
            roe_str = f"{roe_val:.1f}%"

    change_1m = 0.0
    c = candles["c"]
    if len(c) >= 21:
        change_1m = (c[-1] / c[-21] - 1) * 100

    sect_score = 50.0
    for k, v in sector_scores.items():
        if k.lower() in (sector or "").lower() or sector.lower() in k.lower():
            sect_score = v
            break

    return {
        "ticker": symbol,
        "company": str(company)[:30],
        "sector": str(sector),
        "pe": pe,
        "roe": roe_str,
        "change1m": f"{change_1m:+.1f}%",
        "momentum_raw": calc_momentum_score(candles, bm_candles),
        "fundamental_raw": calc_fundamental_score(metrics),
        "sector_raw": sect_score,
    }


async def get_sector_scores(client):
    bm_candles = await get_candles(client, BENCHMARK, days=90)
    if not bm_candles or "c" not in bm_candles or len(bm_candles["c"]) < 21:
        return {k: 50.0 for k in US_SECTOR_ETFS}

    bm_c = bm_candles["c"]
    bm_ret = (bm_c[-1] / bm_c[-21] - 1) * 100

    scores = {}
    for name, etf in US_SECTOR_ETFS.items():
        try:
            res = await get_candles(client, etf, days=90)
            if res and "c" in res and len(res["c"]) >= 21:
                c = res["c"]
                ret = (c[-1] / c[-21] - 1) * 100
                rs = ret - bm_ret
                scores[name] = float(np.clip(50 + rs * 4, 0, 100))
            else:
                scores[name] = 50.0
        except Exception:
            scores[name] = 50.0

    return scores


@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "2.0.0",
        "data_source": "Finnhub",
        "key_configured": bool(FINNHUB_KEY),
    }


@app.get("/api/screen")
async def screen_stocks(
    market: str = Query("US"),
    w_momentum: float = Query(60, ge=0, le=100),
    w_fundamental: float = Query(25, ge=0, le=100),
    w_sector: float = Query(15, ge=0, le=100),
    top_n: int = Query(20, ge=5, le=50),
):
    if not FINNHUB_KEY:
        return JSONResponse(
            {"error": "FINNHUB_KEY not set"},
            status_code=500
        )

    cache_key = f"{market}_{w_momentum}_{w_fundamental}_{w_sector}"
    cached = cache_get(cache_key)
    if cached:
        return JSONResponse({**cached, "from_cache": True})

    async with httpx.AsyncClient() as client:
        sector_scores = await get_sector_scores(client)
        bm_candles = await get_candles(client, BENCHMARK, days=180)

        tasks = [
            analyze_stock(client, t, bm_candles, sector_scores)
            for t in US_TICKERS
        ]
        results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]

    total_w = max(w_momentum + w_fundamental + w_sector, 1)
    for s in valid:
        s["total"] = round(
            (s["momentum_raw"] * w_momentum +
             s["fundamental_raw"] * w_fundamental +
             s["sector_raw"] * w_sector) / total_w, 1
        )
        s["momentum"] = round(s.pop("momentum_raw"), 1)
        s["fundamental"] = round(s.pop("fundamental_raw"), 1)
        s["sector_score"] = round(s.pop("sector_raw"), 1)

    sorted_stocks = sorted(valid, key=lambda x: x["total"], reverse=True)[:top_n]
    for i, s in enumerate(sorted_stocks, 1):
        s["rank"] = i

    sector_list = sorted(
        [
            {
                "name": name,
                "etf": etf,
                "score": round(sector_scores.get(name, 50.0), 1),
                "trend": "강세" if sector_scores.get(name, 50) >= 60
                         else ("중립" if sector_scores.get(name, 50) >= 40 else "약세"),
            }
            for name, etf in US_SECTOR_ETFS.items()
        ],
        key=lambda x: x["score"], reverse=True
    )

    response = {
        "market": "US",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_cache": False,
        "weights": {"momentum": w_momentum, "fundamental": w_fundamental, "sector": w_sector},
        "top20": sorted_stocks,
        "sectors": sector_list,
    }

    cache_set(cache_key, response)
    return JSONResponse(response)

