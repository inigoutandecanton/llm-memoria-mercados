"""Prueba rápida del motor: repartir a partes iguales cada semana."""

import pandas as pd

from llm_memoria.backtest.motor import ejecutar
from llm_memoria.config import cargar_universo
from llm_memoria.datos.precios import descargar_precios


class RepartoIgual:
    nombre = "reparto_igual"

    def decidir(self, precios: pd.DataFrame) -> pd.Series:
        n = len(precios.columns)
        return pd.Series(1 / n, index=precios.columns)


uni = cargar_universo()
p = uni.periodos["pre_corte"]
desde = p.inicio - pd.Timedelta(days=uni.calentamiento_dias)
precios = descargar_precios(uni.tickers, desde, p.fin)

res = ejecutar(RepartoIgual(), precios, inicio=p.inicio, fin=p.fin,
               frecuencia=uni.frecuencia, coste_pb=uni.coste_pb, retraso=uni.retraso)

print(f"Valor final: {res.valor.iloc[-1]:.3f}")
print(f"Rentabilidad total: {res.valor.iloc[-1] - 1:+.1%}")
print(f"Rebalanceos: {len(res.rotacion)}")
print(f"Rotación media: {res.rotacion.mean():.3f}")