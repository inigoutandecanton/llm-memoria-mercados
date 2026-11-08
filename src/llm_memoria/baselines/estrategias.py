"""Estrategias de referencia. Todas usan la misma interfaz que usarán los LLM."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _iguales(columnas: pd.Index, seleccion=None) -> pd.Series:
    """Reparte a partes iguales entre los activos seleccionados (por defecto, todos)."""
    seleccion = columnas if seleccion is None else seleccion
    w = pd.Series(0.0, index=columnas)
    if len(seleccion):
        w[seleccion] = 1.0 / len(seleccion)
    return w


class ComprarYMantener:
    """Equiponderada el primer día y no vuelve a tocar nada."""

    nombre = "comprar_y_mantener"

    def __init__(self) -> None:
        self._invertida = False

    def decidir(self, precios: pd.DataFrame) -> pd.Series | None:
        if self._invertida:
            return None
        self._invertida = True
        return _iguales(precios.columns)


class Equiponderada:
    """1/N, rebalanceada cada semana."""

    nombre = "equiponderada"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        return _iguales(precios.columns)


class Momentum:
    """Compra a partes iguales los top_n activos que más han subido en las últimas `ventana` sesiones."""

    def __init__(self, ventana: int = 60, top_n: int = 5) -> None:
        self.ventana, self.top_n = ventana, top_n
        self.nombre = f"momentum_{ventana}d_top{top_n}"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        if len(precios) <= self.ventana:
            return _iguales(precios.columns)
        rent = precios.iloc[-1] / precios.iloc[-1 - self.ventana] - 1.0
        return _iguales(precios.columns, rent.nlargest(self.top_n).index)


class CruceMedias:
    """Cada activo recibe 1/N si su media corta supera a la larga; si no, esa parte va a liquidez."""

    def __init__(self, corta: int = 50, larga: int = 200) -> None:
        self.corta, self.larga = corta, larga
        self.nombre = f"medias_{corta}_{larga}"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        n = len(precios.columns)
        if len(precios) < self.larga:
            return pd.Series(1.0 / n, index=precios.columns)
        media_corta = precios.iloc[-self.corta:].mean()
        media_larga = precios.iloc[-self.larga:].mean()
        return (media_corta > media_larga).astype(float) / n


class MinimaVarianza:
    """Markowitz de mínima varianza: solo largos y con un peso máximo por activo."""

    def __init__(self, ventana: int = 252, peso_max: float = 0.25) -> None:
        self.ventana, self.peso_max = ventana, peso_max
        self.nombre = f"min_varianza_{ventana}d"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        r = precios.pct_change().dropna().iloc[-self.ventana:]
        n = r.shape[1]
        if len(r) < 60:
            return _iguales(precios.columns)
        cov = r.cov().to_numpy() * 252
        sol = minimize(
            lambda w: w @ cov @ w,
            x0=np.full(n, 1.0 / n),
            jac=lambda w: 2 * cov @ w,
            bounds=[(0.0, max(self.peso_max, 1.0 / n))] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
            method="SLSQP",
        )
        w = sol.x if sol.success else np.full(n, 1.0 / n)
        return pd.Series(np.clip(w, 0, None), index=precios.columns)


class Aleatoria:
    """El 'mono': pesos al azar cada semana. Se repite con muchas semillas."""

    def __init__(self, semilla: int = 0) -> None:
        self.rng = np.random.default_rng(semilla)
        self.nombre = f"aleatoria_{semilla}"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        n = len(precios.columns)
        return pd.Series(self.rng.dirichlet(np.ones(n)), index=precios.columns)


def baselines_por_defecto() -> list:
    return [
        ComprarYMantener(),
        Equiponderada(),
        Momentum(60, 5),
        Momentum(20, 5),
        CruceMedias(50, 200),
        MinimaVarianza(252, 0.25),
    ]