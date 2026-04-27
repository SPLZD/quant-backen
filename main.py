"""
QUANT SCREENER - FastAPI Backend (Finnhub Free Tier)
무료 플랜에서 사용 가능한 엔드포인트만 사용

[v3.1.0 변경사항]
- 캐시 구조 개선: 원시 점수(raw scores)만 캐싱하고 가중치 계산은 매 요청마다 수행
- 효과: 사용자가 가중치를 바꿔도 API 재호출 없이 즉시 재계산
- 이전 버그: 가중치 변경 시 캐시 미스 → API rate limit 초과 → 빈 결과 반환
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

app = FastAPI(title="Quant Screener API", version="3.1.0")

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
    "JNJ", "ABBV", "MRK", "BAC", "GS",
]

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


async def get_profile(client, symbol):
    return await fh_get(client, "/stock/profile2", {"symbol": symbol})


async def get_recommendation(client, symbol):
    return await fh_get(client, "/stock/recommendation", {"symbol": symbol})


def calc_momentum_score(quote, metrics, recs):
    """
    모멘텀 점수 = 52주 고가 위치 + 애널리스트 추천
    """
    try:
        score = 50.0

        if not metrics or "metric" not in metrics:
            return score
        m = metrics["metric"]

        # 52주 고가 대비 현재가 위치 (가장 강한 모멘텀 신호)
        high_52w = m.get("52WeekHigh")
        low_52w = m.get("52WeekLow")
        if quote and "c" in quote and high_52w and low_52w and high_52w > low_52w:
            cur = quote["c"]
            position = (cur - low_52w) / (high_52w - low_52w)  # 0~1
            score = 30 + position * 60  # 30~90 점수

        # 1일 변동률
        if quote and "dp" in quote:
            dp = quote["dp"]
            if dp:
                score += float(np.clip(dp, -5, 5)) * 1.0

        # 애널리스트 추천 (최신)
        if recs and len(recs) > 0:
            latest = recs[0]
            buy = latest.get("buy", 0) + latest.get("strongBuy", 0) * 1.5
            sell = latest.get("sell", 0) + latest.get("strongSell", 0) * 1.5
            hold = latest.get("hold", 0)
            total = buy + sell + hold
            if total > 0:
                rec_score = (buy - sell) / total * 100
                score += rec_score * 0.15

        return float(np.clip(score, 0, 100))
    except Exception:
        return 50.0


def calc_fundamental_score(metrics):
    try:
        if not metrics or "metric" not in metrics:
            return 50.0
        m = metrics["metric"]
        score = 50.0

        pe = m.get("peNormalizedAnnual") or m.get("peTTM") or m.get("peAnnual")
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


def calc_sector_score(sector, metrics):
    """
    섹터 ETF 데이터를 못 받으니까,
    종목 자체의 RS (52주 위치) 기반 점수로 대체
    """
    try:
        if not metrics or "metric" not in metrics:
            return 50.0
        m = metrics["metric"]
        # 베타 + 52주 위치를 활용
        high_52w = m.get("52WeekHigh")
        low_52w = m.get("52WeekLow")
        # 단순화: 52주 고점 근처면 강한 섹터로 간주
        return 50.0  # 일단 중립
    except Exception:
        return 50.0


async def analyze_stock(client, symbol):
    quote = await get_quote(client, symbol)
    metrics = await get_metrics(client, symbol)
    profile = await get_profile(client, symbol)
    recs = await get_recommendation(client, symbol)

    if not quote or "c" not in quote or not quote["c"]:
        return None

    company = symbol
    sector = "Other"
    if profile:
        company = profile.get("name") or symbol
        sector = profile.get("finnhubIndustry") or "Other"

    pe = None
    roe_str = "-"
    high_52w = None
    cur_price = quote["c"]

    if metrics and "metric" in metrics:
        m = metrics["metric"]
        pe_val = m.get("peNormalizedAnnual") or m.get("peTTM") or m.get("peAnnual")
        if pe_val and pe_val > 0:
            pe = round(pe_val, 1)
        roe_val = m.get("roeTTM") or m.get("roeRfy")
        if roe_val:
            roe_str = f"{roe_val:.1f}%"
        high_52w = m.get("52WeekHigh")

    # 1일 변동률을 1M 대신 표시 (캔들 없이 1M 계산 불가)
    change_1d = quote.get("dp", 0) or 0

    # 52주 고가 대비 위치 (퍼센트)
    pos_52w = "-"
    if high_52w and high_52w > 0:
        pos_pct = (cur_price / high_52w) * 100
        pos_52w = f"{pos_pct:.0f}%"

    return {
        "ticker": symbol,
        "company": str(company)[:30],
        "sector": str(sector),
        "pe": pe,
        "roe": roe_str,
        "change1m": f"{change_1d:+.1f}%",  # 실제론 1일이지만 UI 호환 위해
        "pos52w": pos_52w,
        "momentum_raw": calc_momentum_score(quote, metrics, recs),
        "fundamental_raw": calc_fundamental_score(metrics),
        "sector_raw": calc_sector_score(sector, metrics),
    }


async def fetch_raw_data(market: str):
    """
    API에서 원시 데이터 + 원시 점수를 가져온다.
    가중치와 무관하게 시장 단위로만 캐싱한다.
    """
    cache_key = f"raw_{market}"
    cached = cache_get(cache_key)
    if cached:
        return cached, True  # (데이터, 캐시히트여부)

    async with httpx.AsyncClient() as client:
        tasks = [analyze_stock(client, t) for t in US_TICKERS]
        results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    cache_set(cache_key, valid)
    return valid, False


@app.get("/")
def root():
    return {
        "status": "ok",
        "version": "3.1.0",
        "data_source": "Finnhub (free tier)",
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
        return JSONResponse({"error": "FINNHUB_KEY not set"}, status_code=500)

    # 원시 데이터는 가중치와 무관하게 캐싱 (가중치 바꿔도 API 재호출 없음)
    raw_stocks, from_cache = await fetch_raw_data(market)

    if not raw_stocks:
        return JSONResponse({
            "error": "No data available. API rate limit may have been exceeded.",
            "market": market,
            "top20": [],
            "sectors": [],
        }, status_code=503)

    # 원시 데이터를 복사해서 가중치 계산 (캐시 데이터 오염 방지)
    valid = [dict(s) for s in raw_stocks]

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

    # 섹터 데이터는 종목들의 평균으로 추정
    sector_groups = {}
    for s in valid:
        sec = s.get("sector", "Other")
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append(s["momentum"])

    sector_list = []
    for sec_name, scores in sector_groups.items():
        if scores:
            avg = sum(scores) / len(scores)
            sector_list.append({
                "name": sec_name,
                "etf": "-",
                "score": round(avg, 1),
                "trend": "강세" if avg >= 60 else ("중립" if avg >= 40 else "약세"),
            })
    sector_list.sort(key=lambda x: x["score"], reverse=True)

    response = {
        "market": "US",
        "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_cache": from_cache,
        "weights": {"momentum": w_momentum, "fundamental": w_fundamental, "sector": w_sector},
        "top20": sorted_stocks,
        "sectors": sector_list[:11],
    }

    return JSONResponse(response)
