"""
QUANT SCREENER - FastAPI Backend
실시간 yfinance 데이터 + 모멘텀/펀더멘털/섹터 점수 계산
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import time

app = FastAPI(title="Quant Screener API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=8)

# ── 유니버스 ──────────────────────────────────────────────
US_TICKERS = [
    "AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AVGO","AMD","ORCL",
    "CRM","ADBE","QCOM","TXN","INTC","AMAT","LRCX","MU","NOW","SNPS",
    "JPM","BAC","GS","MS","WFC","V","MA","PYPL","AXP","BLK",
    "UNH","LLY","JNJ","ABBV","MRK","PFE","AMGN","GILD","REGN","VRTX",
    "XOM","CVX","COP","EOG","SLB","OXY","MPC","PSX","VLO","HAL",
    "CAT","DE","HON","BA","RTX","LMT","GE","ETN","CTSH","ACN",
    "NEM","FCX","CF","ALB","FSLR","NEE","DUK","COST","WMT","HD",
    "TROW","SCHW","CB","PLD","AMT","EQIX","VICI","NKE","PG","KO",
]

KR_TICKERS = [
    "005930.KS","000660.KS","035420.KS","005380.KS","051910.KS",
    "006400.KS","035720.KS","068270.KS","207940.KS","005490.KS",
    "000270.KS","096770.KS","012330.KS","028260.KS","066570.KS",
    "003550.KS","033780.KS","018260.KS","010130.KS","009150.KS",
    "086790.KS","015760.KS","011200.KS","034730.KS","047050.KS",
    "055550.KS","105560.KS","316140.KS","139480.KS","097950.KS",
    "267250.KS","078930.KS","011070.KS","047810.KS","042700.KS",
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

KR_SECTOR_ETFS = {
    "반도체": "091160.KS",
    "2차전지": "305720.KS",
    "바이오": "244580.KS",
    "자동차": "091180.KS",
    "IT": "266360.KS",
    "금융": "139270.KS",
}

BENCHMARK = {"US": "SPY", "KR": "069500.KS"}

# ── 캐시 ──────────────────────────────────────────────────
_cache: dict = {}
CACHE_TTL = 600


def cache_get(key):
    if key in _cache:
        data, ts = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return data
    return None


def cache_set(key, data):
    _cache[key] = (data, time.time())


# ── 점수 계산 ─────────────────────────────────────────────
def calc_momentum_score(hist: pd.DataFrame, bm_hist: pd.DataFrame) -> float:
    try:
        close = hist["Close"].dropna()
        bm_close = bm_hist["Close"].dropna()
        if len(close) < 20:
            return 0.0
        now = float(close.iloc[-1])
        m1 = (now / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        m3 = (now / float(close.iloc[-63]) - 1) * 100 if len(close) >= 63 else 0.0
        m6 = (now / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else 0.0
        rs = 0.0
        if len(bm_close) >= 21 and len(close) >= 21:
            bm_ret = (float(bm_close.iloc[-1]) / float(bm_close.iloc[-21]) - 1) * 100
            rs = m1 - bm_ret
        raw = m1 * 0.5 + m3 * 0.3 + m6 * 0.1 + rs * 0.1
        return float(np.clip(50 + raw * 1.5, 0, 100))
    except Exception:
        return 0.0


def calc_fundamental_score(info: dict) -> float:
    try:
        score = 50.0
        pe = info.get("trailingPE") or info.get("forwardPE")
        if pe and 0 < float(pe) < 200:
            score += (float(np.clip(100 - abs(float(pe) - 20) * 2, 0, 100)) - 50) * 0.25
        roe = info.get("returnOnEquity")
        if roe:
            score += (min(float(roe) * 500, 100) - 50) * 0.25
        fcf = info.get("freeCashflow")
        mktcap = info.get("marketCap")
        if fcf and mktcap and float(mktcap) > 0:
            score += (min((float(fcf) / float(mktcap)) * 1000, 100) - 50) * 0.2
        growth = info.get("earningsGrowth") or info.get("revenueGrowth")
        if growth:
            score += (min(float(growth) * 300, 100) - 50) * 0.15
        pb = info.get("priceToBook")
        if pb and 0 < float(pb) < 50:
            score += (float(np.clip(100 - abs(float(pb) - 3) * 5, 0, 100)) - 50) * 0.15
        return float(np.clip(score, 0, 100))
    except Exception:
        return 50.0


def calc_sector_score(sector: str, sector_scores: dict) -> float:
    for k, v in sector_scores.items():
        if k.lower() in (sector or "").lower():
            return float(v)
    return 50.0


def fetch_one_stock(ticker: str, bm_hist: pd.DataFrame, sector_scores: dict) -> Optional[dict]:
    try:
        stk = yf.Ticker(ticker)
        hist = stk.history(period="1y")
        if hist.empty or len(hist) < 20:
            return None
        info = stk.info or {}
        company = info.get("longName") or info.get("shortName") or ticker
        sector = info.get("sector") or info.get("industry") or "Other"
        pe = info.get("trailingPE") or info.get("forwardPE")
        roe = info.get("returnOnEquity")
        close = hist["Close"].dropna()
        change_1m = 0.0
        if len(close) >= 21:
            change_1m = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100
        return {
            "ticker": ticker.replace(".KS", "").replace(".KQ", ""),
            "company": str(company)[:30],
            "sector": str(sector),
            "pe": round(float(pe), 1) if pe else None,
            "roe": f"{float(roe)*100:.1f}%" if roe else "–",
            "change1m": f"{change_1m:+.1f}%",
            "momentum_raw": calc_momentum_score(hist, bm_hist),
            "fundamental_raw": calc_fundamental_score(info),
            "sector_raw": calc_sector_score(sector, sector_scores),
        }
    except Exception:
        return None


def fetch_sector_scores(etf_map: dict, benchmark: str) -> dict:
    try:
        bm_close = yf.Ticker(benchmark).history(period="3mo")["Close"].dropna()
        scores = {}
        for sector, etf in etf_map.items():
            try:
                h = yf.Ticker(etf).history(period="3mo")["Close"].dropna()
                if len(h) < 21 or len(bm_close) < 21:
                    scores[sector] = 50.0
                    continue
                ret = (float(h.iloc[-1]) / float(h.iloc[-21]) - 1) * 100
                bm_ret = (float(bm_close.iloc[-1]) / float(bm_close.iloc[-21]) - 1) * 100
                scores[sector] = float(np.clip(50 + (ret - bm_ret) * 3, 0, 100))
            except Exception:
                scores[sector] = 50.0
        return scores
    except Exception:
        return {k: 50.0 for k in etf_map}


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/api/screen")
async def screen_stocks(
    market: str = Query("US", regex="^(US|KR)$"),
    w_momentum: float = Query(60, ge=0, le=100),
    w_fundamental: float = Query(25, ge=0, le=100),
    w_sector: float = Query(15, ge=0, le=100),
    top_n: int = Query(20, ge=5, le=50),
):
    cache_key = f"{market}_{w_momentum}_{w_fundamental}_{w_sector}"
    cached = cache_get(cache_key)
    if cached:
        return JSONResponse({**cached, "from_cache": True})

    loop = asyncio.get_event_loop()
    tickers = US_TICKERS if market == "US" else KR_TICKERS
    etf_map = US_SECTOR_ETFS if market == "US" else KR_SECTOR_ETFS
    benchmark = BENCHMARK[market]

    sector_scores = await loop.run_in_executor(executor, fetch_sector_scores, etf_map, benchmark)
    bm_hist = await loop.run_in_executor(executor, lambda: yf.Ticker(benchmark).history(period="1y"))

    tasks = [
        loop.run_in_executor(executor, fetch_one_stock, t, bm_hist, sector_scores)
        for t in tickers[:40]
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
            for name, etf in etf_map.items()
        ],
        key=lambda x: x["score"], reverse=True
    )

    response = {
        "market": market,
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_cache": False,
        "weights": {"momentum": w_momentum, "fundamental": w_fundamental, "sector": w_sector},
        "top20": sorted_stocks,
        "sectors": sector_list,
    }
    cache_set(cache_key, response)
    return JSONResponse(response)


@app.get("/api/sectors")
async def get_sectors(market: str = Query("US", regex="^(US|KR)$")):
    etf_map = US_SECTOR_ETFS if market == "US" else KR_SECTOR_ETFS
    benchmark = BENCHMARK[market]
    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(executor, fetch_sector_scores, etf_map, benchmark)
    return {
        "market": market,
        "updated": datetime.now().isoformat(),
        "sectors": [
            {
                "name": k, "etf": v,
                "score": round(scores.get(k, 50.0), 1),
                "trend": "강세" if scores.get(k, 50) >= 60 else ("중립" if scores.get(k, 50) >= 40 else "약세"),
            }
            for k, v in etf_map.items()
        ],
    }
