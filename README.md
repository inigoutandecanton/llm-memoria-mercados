# ¿Invierten los LLM o solo recuerdan?

Cuando un LLM "acierta" invirtiendo, ¿está razonando o recordando lo que vio al entrenarse?
Este proyecto lo mide con dos experimentos:

- **A · Test de memorización:** las mismas decisiones con datos reales y anonimizados,
  antes y después de la fecha de corte de cada modelo.
- **B · Arena en vivo:** varios modelos deciden carteras cada semana con datos que no
  pueden conocer, frente a estrategias de referencia.

Estado: **fase 1 completada** (datos, motor de backtest, estrategias de referencia y métricas).

## Diseño contra el sesgo de anticipación

Toda estrategia, sea un baseline o un LLM, implementa `decidir(precios_hasta_t) -> pesos`.
El motor solo le entrega una copia de los precios hasta la fecha de decisión, así que mirar
el futuro no está prohibido: es imposible. Lo decidido al cierre de `t` se ejecuta al cierre
de `t + 1`, con costes de transacción de 10 pb. Los tests lo comprueban explícitamente.

## Resultados de referencia (2022-2023)

| Estrategia | Rentabilidad | Sharpe | Drawdown máx. |
| --- | --- | --- | --- |
| Momentum 60 días (top 5) | +36,5 % | 0,82 | −25,6 % |
| Cruce de medias 50/200 | +12,0 % | 0,68 | −9,1 % |
| Equiponderada semanal | +19,7 % | 0,59 | −21,6 % |
| Comprar y mantener | +14,6 % | 0,49 | −20,9 % |
| Momentum 20 días (top 5) | +15,4 % | 0,42 | −31,2 % |
| Mínima varianza | −1,2 % | 0,01 | −13,9 % |

Cartera aleatoria (200 semillas): Sharpe mediano 0,30; percentil 95: 0,66.

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"
pytest
python experimentos/descargar_datos.py
python experimentos/baselines.py
```

## Estructura

```
config/          activos, periodos y costes
src/llm_memoria/
  datos/         descarga de Yahoo Finance con caché y datos sintéticos
  backtest/      motor y métricas
  baselines/     estrategias de referencia
experimentos/    scripts ejecutables
tests/           pytest
```