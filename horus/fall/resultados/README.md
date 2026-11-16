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
| fase4b_remuestreo_nativo | 8fdb460 | le2i | nativo | LOSO | 96/130 | 73.8% | 2.12 | 2.69 | 29.97 | 0.86 |  |
| fase4b_remuestreo_10 | 8fdb460 | le2i | 10 | LOSO | 89/130 | 68.5% | 2.15 | 2.65 | 31.47 | 0.85 |  |
| fase4c_final_nativo | 8fdb460 | le2i | nativo | LOSO | 85/130 | 65.4% | 2.20 | 2.81 | 14.98 | 0.86 |  |
| fase4c_final_10 | 8fdb460 | le2i | 10 | LOSO | 81/130 | 62.3% | 2.26 | 2.76 | 10.49 | 0.85 |  |
| fase4c_final_8 | 8fdb460 | le2i | 8 | LOSO | 68/130 | 52.3% | 2.33 | 2.93 | 13.49 | 0.85 |  |
| g1_modelo_demo_todo_nativo | 41ab036 | le2i | nativo | modelo_demo_todo.pt | 100/130 | 76.9% | 2.19 | 2.93 | 29.97 | 0.87 |  |
| g1_modelo_demo_todo_10 | 41ab036 | le2i | 10 | modelo_demo_todo.pt | 91/130 | 70.0% | 2.17 | 2.67 | 19.48 | 0.87 |  |
| g1_modelo_despliegue_v2_nativo | 41ab036 | le2i | nativo | modelo_despliegue_v2.pt | 109/130 | 83.8% | 2.12 | 2.58 | 25.47 | 0.88 |  |
| g1_modelo_despliegue_v2_10 | 41ab036 | le2i | 10 | modelo_despliegue_v2.pt | 98/130 | 75.4% | 2.16 | 2.65 | 17.98 | 0.87 |  |
| var_semilla_1 | 41ab036 | le2i | 10 | semilla_1.pt | 100/130 | 76.9% | 2.07 | 2.60 | 34.46 | 0.87 |  |
| var_semilla_2 | 41ab036 | le2i | 10 | semilla_2.pt | 101/130 | 77.7% | 2.08 | 2.52 | 34.46 | 0.88 |  |
| var_semilla_3 | 41ab036 | le2i | 10 | semilla_3.pt | 84/130 | 64.6% | 2.27 | 3.09 | 7.49 | 0.87 |  |
| ensamble4_10 | 41ab036 | le2i | 10 | ensamble x4 | 98/130 | 75.4% | 2.12 | 2.66 | 17.98 | 0.87 |  |
| refactor_10 | 5422165 | le2i | 10 | LOSO | 81/130 | 62.3% | 2.26 | 2.76 | 10.49 | 0.85 |  |
| mej_base_loso_10 | 4ddd0de | le2i | 10 | LOSO | 86/130 | 66.2% | 2.19 | 2.71 | 26.97 | 0.85 | 27/10 linea base en esta PC (CPU). 12 de 18 falsas en stand_up de Home_01: MediaPipe invierte el esqueleto |
| mej_vel0_loso_10 | 4ddd0de | le2i | 10 | LOSO | 86/130 | 66.2% | 2.19 | 2.76 | 26.97 | 0.85 | vel[0] real en el remuestreo (app == entrenamiento, dif 3e-14). Neutro a nivel evento; se queda por correccion |
| mej_desmayo_loso_10 | 4ddd0de | le2i | 10 | LOSO | 95/130 | 73.1% | 1.81 | 2.38 | 128.86 | 0.83 | todos_con_desmayo.npy (caidas estiradas x1.5-3). DESCARTADO: a <=27 falsas/h recall 53% vs 66% base; AUC baja |
| mej_ens3_loso_10 | 4ddd0de | le2i | 10 | LOSO | 89/130 | 68.5% | 2.26 | 2.91 | 22.48 | 0.86 | ensamble 3 semillas por fold. Gana a u=0.5/0.6/0.7 en recall y falsas; a 18/h calibrado empata con la base |
| mej_desplegar_modelo_demo_todo_10 | 4ddd0de | le2i | 10 | modelo_demo_todo.pt | 91/130 | 70.0% | 2.17 | 2.67 | 19.48 | 0.87 | el desplegado hasta el 27/10; reproduce g1_modelo_demo_todo_10 |
| mej_desplegar_modelo_caidas_ens3_10 | 4ddd0de | le2i | 10 | modelo_caidas_ens3.pt | 96/130 | 73.8% | 2.15 | 2.66 | 17.98 | 0.88 | NUEVO desplegado: 18_entrenar.py --semillas 3 con todos.npy. Optimista (vio a todos los sujetos) |
| app_base_x3 | bf1a917 | le2i | 10 | app LOSO x3 | 80/130 | 61.5% | 3.19 | 3.54 | 13.49 | - | cadena de la app; linea base 15/11 |
| app_base_x1 | bf1a917 | le2i | 10 | app LOSO x1 | 79/130 | 60.8% | 3.19 | 3.58 | 11.99 | - | cadena de la app; linea base 15/11 |
| base_x3_ext10 | bf1a917 | le2i | 10 | app LOSO x3 | 92/130 | 70.8% | 3.21 | 3.74 | 13.49 | - | cadena de la app; linea base, videos extendidos 10 s |
| v_sostener | bf1a917 | le2i | 10 | app LOSO x3 | 96/130 | 73.8% | 3.21 | 3.75 | 14.98 | - | cadena de la app; sostener sin pose 10 s |
| v_histeresis | bf1a917 | le2i | 10 | app LOSO x3 | 92/130 | 70.8% | 3.20 | 3.71 | 23.97 | - | cadena de la app; histeresis 0.3 |
| v_pico20 | bf1a917 | le2i | 10 | app LOSO x3 | 98/130 | 75.4% | 3.21 | 3.75 | 16.48 | - | cadena de la app; pico 2.0 |
| upfall_loso_x3_10 | bf1a917 | upfall | 10 | LOSO | 10/12 | 83.3% | 2.52 | 4.01 | 84.09 | 0.52 |  |
| up_base | bf1a917 | upfall | 10 | app LOSO x3 | 9/12 | 75.0% | 3.41 | 3.62 | 49.46 | - | cadena de la app;  |
| up_sostener | bf1a917 | upfall | 10 | app LOSO x3 | 10/12 | 83.3% | 3.46 | 3.83 | 49.46 | - | cadena de la app;  |
| up_pico20 | bf1a917 | upfall | 10 | app LOSO x3 | 9/12 | 75.0% | 3.41 | 3.62 | 49.46 | - | cadena de la app;  |
| v_suav05 | bf1a917 | le2i | 10 | app LOSO x3 | 85/130 | 65.4% | 3.30 | 3.80 | 13.49 | - | cadena de la app; verticalidad mediana 0.5 s |
| up_previa50 | bf1a917 | upfall | 10 | app LOSO x3 | 9/12 | 75.0% | 3.41 | 3.62 | 44.52 | - | cadena de la app;  |
| up_suav05 | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 3.52 | 3.83 | 44.52 | - | cadena de la app;  |
| up_suav05_previa50 | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 3.52 | 3.83 | 44.52 | - | cadena de la app;  |
| up_u060 | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 3.52 | 3.62 | 34.63 | - | cadena de la app;  |
| up_u070 | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 3.54 | 3.70 | 29.68 | - | cadena de la app;  |
| v_suav05_pico20 | bf1a917 | le2i | 10 | app LOSO x3 | 90/130 | 69.2% | 3.31 | 3.78 | 16.48 | - | cadena de la app; mediana 0.5 + pico 2.0 |
| up_parado075 | bf1a917 | upfall | 10 | app LOSO x3 | 7/12 | 58.3% | 3.54 | 7.45 | 39.57 | - | cadena de la app;  |
| up_parado085 | bf1a917 | upfall | 10 | app LOSO x3 | 6/12 | 50.0% | 3.52 | 3.89 | 29.68 | - | cadena de la app;  |
| up_parado075_suav | bf1a917 | upfall | 10 | app LOSO x3 | 7/12 | 58.3% | 3.54 | 4.50 | 24.73 | - | cadena de la app;  |
| v_suav05_sinpico | bf1a917 | le2i | 10 | app LOSO x3 | 91/130 | 70.0% | 3.35 | 3.77 | 16.48 | - | cadena de la app; mediana 0.5, sin pico, torso 0.5 |
| combo_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 103/130 | 79.2% | 3.21 | 3.76 | 17.98 | - | cadena de la app; sostener 10 + pico 2.0 |
| combo_up | bf1a917 | upfall | 10 | app LOSO x3 | 10/12 | 83.3% | 3.46 | 3.83 | 49.46 | - | cadena de la app;  |
| combo_u060_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 98/130 | 75.4% | 3.23 | 3.75 | 14.98 | - | cadena de la app; sostener 10 + pico 2.0 + umbral 0.6 |
| combo_u060_up | bf1a917 | upfall | 10 | app LOSO x3 | 9/12 | 75.0% | 3.54 | 4.00 | 34.63 | - | cadena de la app;  |
| combo_u060_x1_le2i | bf1a917 | le2i | 10 | app LOSO x1 | 96/130 | 73.8% | 3.24 | 3.76 | 13.49 | - | cadena de la app; combo u0.6, 1 semilla (referencia) |
| combo_u060_x1_up | bf1a917 | upfall | 10 | app LOSO x1 | 10/12 | 83.3% | 3.46 | 3.83 | 44.52 | - | cadena de la app;  |
| espejo_le2i | bf1a917 | le2i | 10 | app LOSO x3 +espejo | 99/130 | 76.2% | 3.21 | 3.76 | 17.98 | - | cadena de la app;  |
| loso_app_le2i | bf1a917 | le2i | 10 | LOSO | 88/130 | 67.7% | 2.29 | 3.03 | 16.48 | 0.84 |  |
| quieto15_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 100/130 | 76.9% | 2.77 | 3.33 | 19.48 | - | cadena de la app;  |
| loso_app_upfall | bf1a917 | upfall | 10 | LOSO | 10/12 | 83.3% | 2.52 | 3.62 | 128.61 | 0.41 |  |
| quieto30_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 99/130 | 76.2% | 4.22 | 4.73 | 8.99 | - | cadena de la app;  |
| espejo_upfall | bf1a917 | upfall | 10 | app LOSO x3 +espejo | 9/12 | 75.0% | 3.54 | 4.00 | 34.63 | - | cadena de la app;  |
| quieto15_upfall | bf1a917 | upfall | 10 | app LOSO x3 | 9/12 | 75.0% | 3.09 | 3.51 | 34.63 | - | cadena de la app;  |
| aug_le2i | bf1a917 | le2i | 10 | app LOSO x1 | 94/130 | 72.3% | 3.31 | 3.81 | 14.98 | - | cadena de la app; entrenado simulando la app, 1 semilla |
| aug_up | bf1a917 | upfall | 10 | app LOSO x1 | 9/12 | 75.0% | 3.58 | 4.03 | 44.52 | - | cadena de la app;  |
| quieto30_upfall | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 4.56 | 5.14 | 29.68 | - | cadena de la app;  |
| quieto4_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 100/130 | 76.9% | 5.21 | 5.70 | 4.50 | - | cadena de la app;  |
| quieto4_upfall | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 5.59 | 6.11 | 29.68 | - | cadena de la app;  |
| quieto5_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 98/130 | 75.4% | 6.22 | 6.73 | 4.50 | - | cadena de la app;  |
| quieto5_upfall | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 6.60 | 7.16 | 24.73 | - | cadena de la app;  |
| final_le2i | bf1a917 | le2i | 10 | app LOSO x3 | 100/130 | 76.9% | 5.21 | 5.70 | 4.50 | - | cadena de la app; DEFAULTS NUEVOS (pico 2.0, sostener 10, umbral 0.6, quieto 4, calentamiento 5) |
| final_upfall | bf1a917 | upfall | 10 | app LOSO x3 | 8/12 | 66.7% | 5.59 | 6.02 | 29.68 | - | cadena de la app; DEFAULTS NUEVOS (pico 2.0, sostener 10, umbral 0.6, quieto 4, calentamiento 5) |
| final_le2i_estricto | bf1a917 | le2i | 10 | app LOSO x3 | 37/130 | 28.5% | 5.06 | 5.44 | 4.50 | - | cadena de la app; defaults nuevos, SIN extender (estricto) |
