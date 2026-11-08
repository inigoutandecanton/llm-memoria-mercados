"""Descarga los precios de todos los periodos de config/universo.yaml.

Uso:  python experimentos/descargar_datos.py
"""

import pandas as pd

from llm_memoria.config import cargar_universo
from llm_memoria.datos.precios import descargar_precios


def main() -> None:
    uni = cargar_universo()
    for p in uni.periodos.values():
        desde = p.inicio - pd.Timedelta(days=uni.calentamiento_dias)
        precios = descargar_precios(uni.tickers, desde, p.fin)
        print(f"{p.nombre}: {precios.shape[0]} sesiones x {precios.shape[1]} activos "
              f"({precios.index[0]:%Y-%m-%d} a {precios.index[-1]:%Y-%m-%d})")


if __name__ == "__main__":
    main()