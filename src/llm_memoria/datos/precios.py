"""Descarga de precios desde la API de Yahoo Finance, con caché local en Parquet."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from llm_memoria.config import DIR_DATOS

URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
CABECERAS = {"User-Agent": "Mozilla/5.0"}  # sin esto, Yahoo rechaza la petición


def descargar_ticker(ticker: str, inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.Series:
    """Cierres ajustados (dividendos y splits) de un ticker, indexados por fecha."""
    parametros = {
        "period1": int(inicio.timestamp()),
        "period2": int((fin + pd.Timedelta(days=1)).timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    r = requests.get(URL.format(ticker=ticker), params=parametros, headers=CABECERAS, timeout=30)
    r.raise_for_status()

    grafico = r.json()["chart"]
    if grafico.get("error"):
        raise RuntimeError(f"{ticker}: {grafico['error']}")
    datos = grafico["result"][0]
    if "timestamp" not in datos:
        raise RuntimeError(f"{ticker}: Yahoo no devolvió precios para esas fechas")

    ajustados = datos["indicators"]["adjclose"][0]["adjclose"]
    desfase = datos["meta"]["gmtoffset"]  # segundos de diferencia con UTC de esa bolsa
    fechas = pd.to_datetime([t + desfase for t in datos["timestamp"]], unit="s").normalize()

    serie = pd.Series(ajustados, index=fechas, name=ticker, dtype=float)
    return serie[~serie.index.duplicated(keep="last")]


def descargar_precios(
    tickers: list[str],
    inicio: str | pd.Timestamp,
    fin: str | pd.Timestamp,
    directorio: Path = DIR_DATOS,
    forzar: bool = False,
) -> pd.DataFrame:
    """Una columna por ticker, una fila por fecha. Guarda el resultado en data/."""
    inicio, fin = pd.Timestamp(inicio), pd.Timestamp(fin)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"precios_{inicio:%Y%m%d}_{fin:%Y%m%d}.parquet"

    # --- 1. Si ya lo descargamos antes, lo leemos del disco ---
    if ruta.exists() and not forzar:
        precios = pd.read_parquet(ruta)
        if set(tickers) <= set(precios.columns):
            return precios[tickers]

    # --- 2. Descarga, un ticker cada vez ---
    series = []
    for i, ticker in enumerate(tickers, start=1):
        print(f"  Descargando {ticker} ({i}/{len(tickers)})")
        series.append(descargar_ticker(ticker, inicio, fin))
        time.sleep(0.3)  # pausa breve para no saturar a Yahoo

    # --- 3. Juntar en una tabla ---
    precios = pd.concat(series, axis=1).sort_index()
    precios.index.name = "fecha"

    faltan = [t for t in tickers if precios[t].isna().all()]
    if faltan:
        raise RuntimeError(f"Sin datos para: {faltan}")

    # --- 4. Guardar para la próxima vez ---
    precios.to_parquet(ruta)
    return precios