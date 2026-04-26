from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=8)

@app.get("/")
def root():
    return {"status": "ok", "version": "1.0.0"}

@app.get("/api/test")
def test():
    return {"hello": "world"}
