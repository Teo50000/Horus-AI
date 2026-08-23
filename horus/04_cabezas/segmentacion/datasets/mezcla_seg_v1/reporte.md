# mezcla_seg_v1 -- reporte de armado

Muestras: **80348**  (train 76251 / val 4097)

## Pixeles por clase

| clase | pixeles | % | imagenes que la contienen |
|---|---:|---:|---:|
| 0 fondo | 17,022,438,636 | 83.56% | 80,306 |
| 1 agua | 2,298,225,786 | 11.28% | 22,258 |
| 2 humo | 1,010,971,921 | 4.96% | 22,314 |
| 3 fuego | 40,213,701 | 0.20% | 27,579 |

## Muestras por fuente

| fuente | muestras |
|---|---:|
| fire_seg_killa | 27,460 |
| multinatsmoke | 22,318 |
| cocostuff | 19,170 |
| atlantis | 5,188 |
| flood_masks | 3,401 |
| riwa | 1,632 |
| flood_semantic | 663 |
| flood_area | 290 |
| bowfire | 226 |

## Avisos

- Desbalance de pixeles >20x entre clases positivas. `entrenar_segmentacion_cuda.py` lo compensa con pesos, pero conviene topear la fuente dominante con --tope-fuente.
