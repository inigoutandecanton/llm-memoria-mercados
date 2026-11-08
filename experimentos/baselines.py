"""Ejecuta todos los baselines y el 'mono' aleatorio sobre un periodo.

Uso:
  python experimentos/baselines.py                         # periodo pre_corte
  python experimentos/baselines.py --periodo post_corte
  python experimentos/baselines.py --n-aleatorias 50       # más rápido
"""

import argparse

import pandas as pd

from llm_memoria.backtest.metricas import metricas, tabla_resumen
from llm_memoria.backtest.motor import ejecutar
from llm_memoria.baselines import Aleatoria, baselines_por_defecto
from llm_memoria.config import DIR_RESULTADOS, cargar_universo
from llm_memoria.datos.precios import descargar_precios


def main() -> None:
    # --- 1. Leer opciones de la línea de comandos ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--periodo", default="pre_corte")
    ap.add_argument("--n-aleatorias", type=int, default=200)
    args = ap.parse_args()

    # --- 2. Cargar configuración y precios ---
    uni = cargar_universo()
    p = uni.periodos[args.periodo]
    desde = p.inicio - pd.Timedelta(days=uni.calentamiento_dias)
    precios = descargar_precios(uni.tickers, desde, p.fin)
    opciones = dict(inicio=p.inicio, fin=p.fin, frecuencia=uni.frecuencia,
                    coste_pb=uni.coste_pb, retraso=uni.retraso)

    # --- 3. Ejecutar los baselines ---
    resultados = [ejecutar(e, precios, **opciones) for e in baselines_por_defecto()]
    tabla = tabla_resumen(resultados)

    # --- 4. Ejecutar el mono muchas veces ---
    sharpes = pd.Series([
        metricas(ejecutar(Aleatoria(s), precios, **opciones).valor)["sharpe"]
        for s in range(args.n_aleatorias)
    ])

    # --- 5. Mostrar resultados ---
    pd.set_option("display.width", 160)
    print(f"\n=== Baselines · {p.nombre} · {p.inicio:%Y-%m-%d} a {p.fin:%Y-%m-%d} ===\n")
    columnas = ["rentabilidad_total", "cagr", "volatilidad", "sharpe", "max_drawdown", "rotacion_media"]
    print(tabla[columnas].round(3).to_string())
    print(f"\nMono aleatorio ({args.n_aleatorias} semillas), Sharpe: "
          f"p5={sharpes.quantile(0.05):.2f}  mediana={sharpes.median():.2f}  "
          f"p95={sharpes.quantile(0.95):.2f}")

    # --- 6. Guardar ---
    DIR_RESULTADOS.mkdir(exist_ok=True)
    tabla.to_csv(DIR_RESULTADOS / f"baselines_{p.nombre}.csv")
    pd.concat([r.valor for r in resultados], axis=1).to_csv(DIR_RESULTADOS / f"curvas_{p.nombre}.csv")
    sharpes.to_csv(DIR_RESULTADOS / f"aleatorias_sharpe_{p.nombre}.csv", index=False)
    print(f"\nGuardado en {DIR_RESULTADOS}")


if __name__ == "__main__":
    main()