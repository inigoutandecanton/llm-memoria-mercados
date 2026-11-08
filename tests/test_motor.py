import numpy as np
import pandas as pd
import pytest

from llm_memoria.backtest.motor import ejecutar, validar_pesos
from llm_memoria.baselines import ComprarYMantener, Equiponderada
from llm_memoria.datos.sinteticos import precios_gbm


# --- Estrategias de prueba ---
class Espia:
    """No invierte; solo apunta qué fechas ve en cada decisión."""

    nombre = "espia"

    def __init__(self):
        self.ultimas_fechas_vistas = []
        self.fechas_completas = None

    def decidir(self, precios):
        self.ultimas_fechas_vistas.append(precios.index[-1])
        return None


class TodoEnBEnLaFecha:
    """Mete todo en el activo B, pero solo el día indicado."""

    nombre = "todo_en_b"

    def __init__(self, fecha):
        self.fecha = fecha

    def decidir(self, precios):
        if precios.index[-1] == self.fecha:
            return pd.Series({"A": 0.0, "B": 1.0})
        return None


def precios_con_salto(dia_salto: int) -> pd.DataFrame:
    """Dos activos planos a 100; B salta a 200 en la sesión dia_salto."""
    fechas = pd.bdate_range("2024-01-01", periods=10)
    b = np.full(10, 100.0)
    b[dia_salto:] = 200.0
    return pd.DataFrame({"A": np.full(10, 100.0), "B": b}, index=fechas)


# --- Tests de no mirar el futuro ---
def test_la_estrategia_nunca_ve_mas_alla_de_la_fecha_de_decision():
    precios = precios_gbm(3, "2022-01-01", "2022-12-31")
    espia = Espia()
    ejecutar(espia, precios)
    assert len(espia.ultimas_fechas_vistas) > 40  # unas 52 semanas
    # Las fechas vistas avanzan semana a semana y nunca llegan de golpe al final.
    assert espia.ultimas_fechas_vistas == sorted(espia.ultimas_fechas_vistas)
    assert espia.ultimas_fechas_vistas[0] == precios.index[0]


def test_decidir_hoy_no_aprovecha_el_salto_de_manana():
    precios = precios_con_salto(dia_salto=5)
    dia_antes_del_salto = precios.index[4]
    res = ejecutar(TodoEnBEnLaFecha(dia_antes_del_salto), precios,
                   frecuencia="D", coste_pb=0, retraso=1)
    assert res.valor.iloc[-1] == pytest.approx(1.0)


def test_sin_retraso_si_lo_aprovecharia():
    precios = precios_con_salto(dia_salto=5)
    dia_antes_del_salto = precios.index[4]
    res = ejecutar(TodoEnBEnLaFecha(dia_antes_del_salto), precios,
                   frecuencia="D", coste_pb=0, retraso=0)
    assert res.valor.iloc[-1] == pytest.approx(2.0)


# --- Tests de contabilidad ---
def test_comprar_y_mantener_sigue_a_los_precios():
    precios = precios_gbm(4, "2022-01-01", "2022-12-31", semilla=3)
    res = ejecutar(ComprarYMantener(), precios, coste_pb=0, retraso=0)
    esperado = (precios.iloc[-1] / precios.iloc[0]).mean()
    assert res.valor.iloc[-1] == pytest.approx(esperado)


def test_la_primera_compra_cuesta_exactamente_10_pb():
    precios = pd.DataFrame({"A": [100.0] * 5}, index=pd.bdate_range("2024-01-01", periods=5))
    res = ejecutar(ComprarYMantener(), precios, coste_pb=10, retraso=0)
    assert res.valor.iloc[-1] == pytest.approx(0.999)


def test_los_costes_reducen_la_rentabilidad():
    precios = precios_gbm(4, "2022-01-01", "2022-06-30")
    sin_costes = ejecutar(Equiponderada(), precios, coste_pb=0).valor.iloc[-1]
    con_costes = ejecutar(Equiponderada(), precios, coste_pb=10).valor.iloc[-1]
    assert con_costes < sin_costes


def test_validar_pesos_quita_cortos_y_limita_la_suma():
    columnas = pd.Index(["A", "B", "C"])
    w = validar_pesos(pd.Series({"A": 2.0, "B": -1.0, "Z": 5.0}), columnas)
    assert w.tolist() == [1.0, 0.0, 0.0]  # B negativo -> 0, Z no existe, A reescalado a 1
    w = validar_pesos(pd.Series({"A": 0.3, "B": 0.2}), columnas)
    assert w.sum() == pytest.approx(0.5)  # el 50 % restante queda en liquidez