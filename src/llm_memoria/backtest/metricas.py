"""Métricas de rendimiento y de comportamiento de una cartera."""

from __future__ import annotations

import numpy as np
import pandas as pd

from llm_memoria.backtest.motor import Resultado

SESIONES_ANO = 252


def max_drawdown(valor: pd.Series) -> float:
    """Peor caída desde un máximo anterior (número negativo)."""
    return float((valor / valor.cummax() - 1.0).min())


def metricas(valor: pd.Series, rf_anual: float = 0.0, periodos: int = SESIONES_ANO) -> dict:
    r = valor.pct_change().dropna()
    anos = len(r) / periodos
    total = valor.iloc[-1] / valor.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / anos) - 1.0

    exceso = r - rf_anual / periodos
    sd = r.std(ddof=1)
    vol = sd * np.sqrt(periodos)
    sharpe = exceso.mean() / sd * np.sqrt(periodos) if sd > 0 else np.nan

    abajo = np.sqrt(np.mean(np.minimum(exceso, 0.0) ** 2)) * np.sqrt(periodos)
    sortino = exceso.mean() * periodos / abajo if abajo > 0 else np.nan

    mdd = max_drawdown(valor)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan

    return {
        "rentabilidad_total": total,
        "cagr": cagr,
        "volatilidad": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
    }


def metricas_comportamiento(res: Resultado) -> dict:
    """Cómo opera la estrategia, no solo cuánto gana."""
    return {
        "rotacion_media": float(res.rotacion.mean()) if len(res.rotacion) else 0.0,
        "n_rebalanceos": int(len(res.rotacion)),
        "herfindahl_medio": float((res.pesos**2).sum(axis=1).mean()),
        "exposicion_media": float(res.pesos.sum(axis=1).mean()),
    }


def tabla_resumen(resultados: list[Resultado], rf_anual: float = 0.0) -> pd.DataFrame:
    """Una fila por estrategia, ordenadas por Sharpe."""
    filas = {
        res.nombre: {**metricas(res.valor, rf_anual), **metricas_comportamiento(res)}
        for res in resultados
    }
    return pd.DataFrame(filas).T.sort_values("sharpe", ascending=False)