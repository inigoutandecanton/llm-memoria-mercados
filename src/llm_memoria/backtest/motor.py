"""Motor de backtest sin fugas de futuro.

Una estrategia es cualquier objeto con un método decidir(precios_hasta_t) -> pesos.
El motor solo le entrega una COPIA de los precios hasta la fecha de decisión.

Convenciones:
- Pesos = fracciones del capital, solo largos. Lo no invertido queda en liquidez (rentabilidad 0).
- Lo decidido al cierre de t se ejecuta al cierre de t + retraso y empieza a rentar al día siguiente.
- Entre rebalanceos los pesos se mueven con los precios.
- Coste = rotación (suma de |cambios de peso|) x coste_pb / 10.000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


# --- 1. Qué es una estrategia y qué devuelve el motor ---
class Estrategia(Protocol):
    nombre: str

    def decidir(self, precios: pd.DataFrame) -> pd.Series | None:
        """Pesos objetivo con los precios hasta hoy, o None para no operar."""
        ...


@dataclass
class Resultado:
    nombre: str
    valor: pd.Series          # valor de la cartera cada día (empieza en 1)
    pesos: pd.DataFrame       # pesos que se tienen al final de cada día
    rotacion: pd.Series       # cuánto se movió la cartera en cada rebalanceo
    decisiones: pd.DataFrame  # pesos que pidió la estrategia en cada decisión

    @property
    def rentabilidades(self) -> pd.Series:
        return self.valor.pct_change().dropna()


# --- 2. Funciones auxiliares ---
def validar_pesos(pesos: pd.Series, columnas: pd.Index) -> np.ndarray:
    """Alinea con el universo, quita cortos y vacíos, y limita la suma a 1."""
    w = pd.Series(pesos, dtype=float).reindex(columnas).fillna(0.0).clip(lower=0.0)
    total = w.sum()
    if total > 1.0:
        w = w / total
    return w.to_numpy()


def fechas_de_decision(indice: pd.DatetimeIndex, frecuencia: str = "W-FRI") -> pd.DatetimeIndex:
    """Última sesión disponible de cada semana."""
    s = pd.Series(indice, index=indice)
    return pd.DatetimeIndex(s.resample(frecuencia).last().dropna().to_numpy())


# --- 3. El motor ---
def ejecutar(
    estrategia: Estrategia,
    precios: pd.DataFrame,
    inicio: str | pd.Timestamp | None = None,
    fin: str | pd.Timestamp | None = None,
    frecuencia: str = "W-FRI",
    coste_pb: float = 10.0,
    retraso: int = 1,
) -> Resultado:
    # 3a. Preparar precios y fechas
    precios = precios.sort_index().ffill().dropna(how="any")
    inicio = pd.Timestamp(inicio) if inicio is not None else precios.index[0]
    fin = pd.Timestamp(fin) if fin is not None else precios.index[-1]
    dias = precios.index[(precios.index >= inicio) & (precios.index <= fin)]
    if len(dias) < 2:
        raise ValueError("El periodo tiene menos de dos sesiones.")

    fechas = fechas_de_decision(dias, frecuencia).union([dias[0]])
    columnas = precios.columns
    posicion = {fecha: i for i, fecha in enumerate(dias)}

    # 3b. Pedir las decisiones, enseñando solo el pasado
    ejecuciones: dict[int, np.ndarray] = {}
    registro: dict[pd.Timestamp, np.ndarray] = {}
    for t in fechas:
        pasado = precios.loc[:t].copy()
        objetivo = estrategia.decidir(pasado)
        if objetivo is None:
            continue
        w = validar_pesos(objetivo, columnas)
        registro[t] = w
        dia_ejecucion = posicion[t] + retraso
        if dia_ejecucion < len(dias):
            ejecuciones[dia_ejecucion] = w

    # 3c. Simular día a día
    rent = precios.pct_change().loc[dias].fillna(0.0).to_numpy()
    coste = coste_pb / 10_000
    w = np.zeros(len(columnas))   # empezamos todo en liquidez
    capital = 1.0
    valores = np.empty(len(dias))
    pesos = np.empty((len(dias), len(columnas)))
    rotaciones: dict[pd.Timestamp, float] = {}

    for d in range(len(dias)):
        if d > 0:
            crecimiento = 1.0 + w @ rent[d]
            w = w * (1.0 + rent[d]) / crecimiento
            capital *= crecimiento
        if d in ejecuciones:
            objetivo = ejecuciones[d]
            rot = float(np.abs(objetivo - w).sum())
            capital *= 1.0 - rot * coste
            w = objetivo.copy()
            rotaciones[dias[d]] = rot
        valores[d] = capital
        pesos[d] = w

    return Resultado(
        nombre=estrategia.nombre,
        valor=pd.Series(valores, index=dias, name=estrategia.nombre),
        pesos=pd.DataFrame(pesos, index=dias, columns=columnas),
        rotacion=pd.Series(rotaciones, name="rotacion", dtype=float),
        decisiones=pd.DataFrame.from_dict(registro, orient="index", columns=columnas),
    )