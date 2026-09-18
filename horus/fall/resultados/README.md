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
| A_tol1.0_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 86/130 | 66.2% | 2.23 | 2.79 | 22.48 | 0.86 | tolerancia=1.0 |
| A_tol1.5_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 94/130 | 72.3% | 2.21 | 2.78 | 26.97 | 0.86 | tolerancia=1.5 |
| A_tol2.0_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 97/130 | 74.6% | 2.20 | 2.77 | 29.97 | 0.86 | tolerancia=2.0 |
| A_tol3.0_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 98/130 | 75.4% | 2.20 | 2.77 | 31.47 | 0.86 | tolerancia=3.0 |
| C_pico_tol0.6_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 63/130 | 48.5% | 2.21 | 2.75 | 7.49 | 0.86 | tolerancia=0.6 |
| C_pico_tol2.0_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 76/130 | 58.5% | 2.20 | 2.77 | 23.97 | 0.86 | tolerancia=2.0 |
| C2_desac_tol0.6_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 74/130 | 56.9% | 2.24 | 2.79 | 7.49 | 0.86 | tolerancia=0.6 |
| C2_desac_tol2.0_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 90/130 | 69.2% | 2.21 | 2.78 | 23.97 | 0.86 | tolerancia=2.0 |
| C3_nobypass_f3.0_tol2.0 | 5c438f7 | le2i | nativo | LOSO | 90/130 | 69.2% | 2.21 | 2.78 | 19.48 | 0.86 | tolerancia=2.0, factor_pico=3.0, sin bypass |
| C3_nobypass_f2.5_tol2.0 | 5c438f7 | le2i | nativo | LOSO | 93/130 | 71.5% | 2.20 | 2.78 | 19.48 | 0.86 | tolerancia=2.0, factor_pico=2.5, sin bypass |
| C3_nobypass_f2.0_tol2.0 | 5c438f7 | le2i | nativo | LOSO | 96/130 | 73.8% | 2.20 | 2.77 | 25.47 | 0.86 | tolerancia=2.0, factor_pico=2.0, sin bypass |
| C3_nobypass_f2.5_tol0.6 | 5c438f7 | le2i | nativo | LOSO | 76/130 | 58.5% | 2.21 | 2.73 | 5.99 | 0.86 | tolerancia=0.6, factor_pico=2.5, sin bypass |
| C3_nobypass_f2.5_tol1.0 | 5c438f7 | le2i | nativo | LOSO | 82/130 | 63.1% | 2.22 | 2.78 | 8.99 | 0.86 | tolerancia=1.0, factor_pico=2.5, sin bypass |
| C3_nobypass_f2.5_tol1.5 | 5c438f7 | le2i | nativo | LOSO | 91/130 | 70.0% | 2.20 | 2.78 | 16.48 | 0.86 | tolerancia=1.5, factor_pico=2.5, sin bypass |
| C3_nobypass_f2.5_tol1.5_10fps | 5c438f7 | le2i | 10 | LOSO | 69/130 | 53.1% | 2.85 | 3.44 | 7.49 | 0.85 | tolerancia=1.5, factor_pico=2.5, sin bypass |
| C3_nobypass_f2.5_tol2.0_10fps | 5c438f7 | le2i | 10 | LOSO | 77/130 | 59.2% | 2.83 | 3.44 | 10.49 | 0.85 | tolerancia=2.0, factor_pico=2.5, sin bypass |
| fase4a_defaults_loso_nativo | 5c438f7 | le2i | nativo | LOSO | 93/130 | 71.5% | 2.20 | 2.78 | 19.48 | 0.86 |  |
