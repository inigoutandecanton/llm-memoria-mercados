"""Precios sintéticos (movimiento browniano geométrico) para tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def precios_gbm(
    n_activos: int = 10,
    inicio: str = "2020-01-01",
    fin: str = "2023-12-31",
    mu_anual: float = 0.07,
    vol_anual: float = 0.25,
    correlacion: float = 0.3,
    semilla: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range(inicio, fin, name="fecha")
    dt = 1 / 252

    cov = np.full((n_activos, n_activos), correlacion) + (1 - correlacion) * np.eye(n_activos)
    z = rng.multivariate_normal(np.zeros(n_activos), cov, size=len(fechas))
    mus = rng.normal(mu_anual, 0.05, n_activos)
    vols = vol_anual * rng.uniform(0.6, 1.4, n_activos)

    log_r = (mus - 0.5 * vols**2) * dt + vols * np.sqrt(dt) * z
    log_r[0] = 0.0
    precios = 100 * np.exp(np.cumsum(log_r, axis=0))
    return pd.DataFrame(precios, index=fechas, columns=[f"ACT{i:02d}" for i in range(n_activos)])