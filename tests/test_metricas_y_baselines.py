import numpy as np
import pandas as pd
import pytest

from llm_memoria.backtest.metricas import max_drawdown, metricas
from llm_memoria.backtest.motor import ejecutar
from llm_memoria.baselines import Aleatoria, baselines_por_defecto
from llm_memoria.datos.sinteticos import precios_gbm


def test_max_drawdown():
    valor = pd.Series([1.0, 1.2, 0.9, 1.1, 1.5])
    assert max_drawdown(valor) == pytest.approx(0.9 / 1.2 - 1)  # de 1,2 a 0,9: −25 %


def test_crecimiento_constante_no_tiene_drawdown():
    valor = pd.Series(1.0005 ** np.arange(253))  # sube un 0,05 % cada día durante un año
    m = metricas(valor)
    assert m["max_drawdown"] == pytest.approx(0.0)
    assert m["cagr"] == pytest.approx(1.0005**252 - 1)


def test_todos_los_baselines_dan_pesos_validos():
    precios = precios_gbm(8, "2021-01-01", "2022-12-31")
    for estrategia in baselines_por_defecto() + [Aleatoria(1)]:
        res = ejecutar(estrategia, precios, inicio="2022-01-01")
        assert res.valor.notna().all(), estrategia.nombre
        assert (res.pesos.to_numpy() >= -1e-9).all(), estrategia.nombre
        assert (res.pesos.sum(axis=1) <= 1 + 1e-9).all(), estrategia.nombre