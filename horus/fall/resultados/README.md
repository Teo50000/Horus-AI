# Resultados a nivel evento (17_evaluar_evento.py)

Una fila por corrida. Detalle completo en `evento_{tag}.json`.

| tag | git | origen | fps | modelo | eventos | recall | lat. med (s) | lat. p90 (s) | falsas/h | AUC vent. | notas |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chk_regla_persistencia_s0 | 25294f9 | le2i | nativo | LOSO | 81/130 | 62.3% | 2.21 | 2.69 | 32.96 | 0.86 | sin pico, descarte 1.5s |
| chk_demo_s0 | 25294f9 | le2i | nativo | modelo_demo_s0.pt | 89/130 | 68.5% | 2.09 | 2.63 | 46.45 | 0.88 | sin pico, descarte 1.5s |
| base_desplegado_nativo | 25294f9 | le2i | nativo | modelo_demo_todo.pt | 93/130 | 71.5% | 2.20 | 2.84 | 26.97 | 0.87 |  |
| base_desplegado_10fps | 25294f9 | le2i | 10 | modelo_demo_todo.pt | 71/130 | 54.6% | 2.68 | 3.50 | 14.98 | 0.85 |  |
| base_loso_nativo | 25294f9 | le2i | nativo | LOSO | 79/130 | 60.8% | 2.22 | 2.77 | 14.98 | 0.86 |  |
| base_loso_10fps | 25294f9 | le2i | 10 | LOSO | 63/130 | 48.5% | 2.88 | 3.41 | 10.49 | 0.85 |  |
| base_loso_nativo_sinpico | 25294f9 | le2i | nativo | LOSO | 81/130 | 62.3% | 2.21 | 2.69 | 32.96 | 0.86 | sin pico |
