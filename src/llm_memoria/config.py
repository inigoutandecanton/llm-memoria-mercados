"""Carga de la configuración del proyecto (carpeta config/)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

# --- Bloque 1: rutas del proyecto ---
RAIZ = Path(__file__).resolve().parents[2]
DIR_CONFIG = RAIZ / "config"
DIR_DATOS = RAIZ / "data"
DIR_RESULTADOS = RAIZ / "resultados"


# --- Bloque 2: la forma de los datos ---
@dataclass(frozen=True)
class Periodo:
    nombre: str
    inicio: pd.Timestamp
    fin: pd.Timestamp


@dataclass(frozen=True)
class Universo:
    tickers: list[str]
    periodos: dict[str, Periodo]
    frecuencia: str
    calentamiento_dias: int
    coste_pb: float
    retraso: int


# --- Bloque 3: la función que lee el YAML ---
def cargar_universo(ruta: Path | None = None) -> Universo:
    ruta = ruta or DIR_CONFIG / "universo.yaml"
    with open(ruta, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    periodos = {
        nombre: Periodo(nombre, pd.Timestamp(p["inicio"]), pd.Timestamp(p["fin"]))
        for nombre, p in cfg["periodos"].items()
    }
    return Universo(
        tickers=list(cfg["tickers"]),
        periodos=periodos,
        frecuencia=cfg.get("frecuencia", "W-FRI"),
        calentamiento_dias=int(cfg.get("calentamiento_dias", 400)),
        coste_pb=float(cfg.get("coste_pb", 10)),
        retraso=int(cfg.get("retraso", 1)),
    )