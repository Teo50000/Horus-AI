# Datasets para la cabeza de objetos — HORUS

Verificado el 18/08/2026. Cada fila dice licencia y si sirve para producto
comercial, porque eso es lo que después no se puede deshacer.

**Leyenda de licencia:** ✅ comercial OK · ⚠️ zona gris o hay que pedir permiso ·
❌ no comercial (sirve para prototipar, NO para el modelo que se despliega)

---

## ⚠️ Dos cambios obligatorios respecto de `datasets/LEEME.txt`

**1. CrowdHuman no se puede usar en un producto.** Los términos de Megvii dicen
*"You will use the data only for non-commercial research and educational
purposes"*, y la ficha en Hugging Face lo confirma como CC-BY-NC-4.0. Que
alguien lo haya resubido a Roboflow con etiqueta "CC BY 4.0" no relicencia
nada — el que sube no es dueño de los derechos. Además CrowdHuman **no es
CCTV**: son fotos de Google Images de multitudes.

**2. OD-WeaponDetection (Granada) solo no alcanza — está medido.** El paper de
Monash (APSIPA 2019) entrenó con ese dataset y evaluó sobre objetos chicos:
**0 % mAP en small objects**. Son fotos web y frames de películas (James Bond,
Pulp Fiction, Mr. Bean literalmente están en el set de test). A escala CCTV
no transfiere. Sirve, pero como complemento, nunca como base.

---

## Resumen: qué bajar para cada clase

| # | Clase | Fuente principal | Licencia | Complemento |
|---|---|---|---|---|
| 0 | humo | D-Fire | ✅ CC0 | Pyro-SDIS (cámara fija) |
| 1 | llama | D-Fire | ✅ CC0 | MS-FSDB |
| 2 | persona | metraje propio + MEVA | ✅ CC BY 4.0 | Open Images filtrado |
| 3 | pistola | Monash Guns (MGD) | ✅ MIT | ACF (pedir) + grabar propio |
| 4 | cuchillo | Open Images + ACF | ✅ CC BY | grabar propio |
| 5 | celular | Sohas smartphone | ⚠️ CC BY-SA | COCO / Open Images |
| 6 | paquete | package-at-front-door | ✅ MIT | boxes-iqjbg |

---

## 0 y 1 · HUMO y LLAMA

### ★ D-Fire — la base, sin discusión
https://github.com/gaiasd/DFireDataset

- **21.527 imágenes** · 11.865 cajas de humo · 14.692 de fuego
- **9.838 imágenes vacías** (sin objetos) — oro puro para bajar falsos positivos
- **YOLO nativo, `0 = smoke`, `1 = fire`** → coincide exacto con tus clases 0 y 1.
  **No hay que remapear nada.**
- ✅ **CC0-1.0** (dominio público, comercial explícito)
- Mirror Kaggle: https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

⚠️ **Deduplicar antes de partir train/val.** El paper de DetectiumFire midió
3.201 duplicados por similitud CNN y 883 por pHash. Si partís sin deduplicar
hay fuga entre train y val y el mAP te miente.

### ★ Pyro-SDIS — lo que a D-Fire le falta: cámara fija y humo tenue
https://huggingface.co/datasets/pyronear/pyro-sdis

- **33.636 imágenes** · 31.975 cajas · ~3,3 GB
- Cámaras **FIJAS** de torres de bomberos franceses (SDIS). Humo real, tenue, lejano.
- YOLO, **una sola clase `0 = smoke`** → remapear a tu `0`. Sin conflicto.
- ✅ **Apache-2.0**

```python
from datasets import load_dataset
ds = load_dataset("pyronear/pyro-sdis")   # image · annotations (texto YOLO) · image_name
```

### Negativos duros — para que no te alerte con cada atardecer
- **FireAndSmoke** (clase `other`: atardeceres, faroles, reflejos, balizas, pantallas)
  - part1 (21.118 imgs): https://universe.roboflow.com/catargiuconstantin/firesmokedataset/dataset/2
  - part2 (6.176 imgs): https://universe.roboflow.com/catargiuconstantin2/firesmokenewdataset/dataset/1
  - ✅ CC BY 4.0 · 15.812 día + 7.158 noche
- **SAINetset v8.0** (AlterMundi, Córdoba): https://huggingface.co/datasets/SAINetset/SAINetset_v8.0
  - ✅ CC BY 4.0 · 5.596 imgs de las cuales **5.351 son negativos de nodos de
    vigilancia reales desplegados**. Es exactamente el dato que baja falsas alarmas.

### Humo incipiente (lo que importa para prevenir)
- **Nemo**: https://github.com/SayBender/Nemo · mirror https://datasetninja.com/nemo
  - ✅ Apache-2.0 · 2.934 imgs · humo etiquetado por densidad **low/mid/high**
  - Detectó 97,9 % de incendios en estadio incipiente. Objetos de hasta **24×26 px**.
- **MS-FSDB**: https://github.com/xiaoyihan6/ms-fsdb — ✅ MIT · 12.586 imgs (3.603 pos / 8.983 neg)

### ❌ No usar
| Dataset | Motivo |
|---|---|
| AI For Mankind wildfire-smoke | CC BY-**NC**-SA 4.0 |
| DetectiumFire | CC BY-NC-SA + "academic use only" |
| Smoke100k | prohíbe explícitamente uso comercial |
| FASDD | ⚠️ licencia **sin confirmar**; el "CC BY 4.0" que se ve es del *paper*, no del dato. Si lo querés, pedí la licencia por escrito a los autores primero |
| FIgLib / HPWREN | cajas solo para 144 incendios, no empaquetadas; licencia contradictoria |

**Dato que confirma tu propia nota técnica:** Nemo mide objetos de humo de
24×26 px en imágenes de 3072×2048. A 320 px de entrada eso es sub-píxel.
**No bajes de 640 si querés detección temprana** — tu punto 7 del orden de
optimización está bien fundado.

---

## 2 · PERSONA

La conclusión honesta: **no existe un dataset grande, con licencia permisiva,
de CCTV real y con cajas exhaustivas.** Todo lo que tiene la vista correcta
(MOT20, CityPersons, LLVIP, SOMPT22) es no comercial; todo lo comercialmente
usable (Open Images, COCO) son fotos web. Hay que armarlo en tres capas.

### ★ Capa 1 — tu propio metraje RTSP (prioridad máxima, ~1.500-2.500 imgs)
Es la única fuente con tu altura de montaje, tu lente, tu bitrate, tus
artefactos de compresión y tu conmutación día/IR. **1.000 frames en dominio
valen más que 50.000 fotos web.** Riesgo legal: cero, el metraje es tuyo.

Receta: muestreá frames de todas las cámaras (día / noche / conmutación IR /
lluvia / pasillo vacío), pseudo-etiquetá con **RF-DETR** (Apache 2.0, comercial
libre: https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/) y corregí
a mano. ~2 días de trabajo para 2.000 frames.

**Bonus:** este mismo corpus te sirve para calibrar el INT8 de TensorRT, que ya
figura como pendiente en `LEEME_CUDA.txt`. Se junta una sola vez.

### ★ Capa 2 — MEVA (~1.000-1.500 imgs)
https://mevadata.org/

- **328 h de CCTV fijo real**, 38 cámaras, 33 puntos de vista, interior y exterior
- **Cámaras térmicas IR co-localizadas** — cubre el nocturno
- Personas de 10-200 px de altura: tu régimen de escala
- ✅ **CC BY 4.0** (permite derivados comerciales, solo hay que atribuir)

⚠️ **No uses el `.geom.yml` como ground truth.** MEVA solo anota a los actores
que participan en actividades anotadas, no a toda la gente del frame. Usarlo
tal cual le enseña al modelo a NO detectar personas. Tratalo como corpus de
imágenes CCTV legalmente limpias y pseudo-etiquetá igual que la capa 1.

### Capa 3 — Open Images V7, subset Person (~1.000-1.500 imgs)
Clase `Person` = MID **`/m/01g317`** · ✅ anotaciones CC BY 4.0, imágenes CC BY 2.0

Filtros obligatorios:
- `IsGroupOf == 0` → si no, metés una caja gigante sobre una multitud entera
- `IsDepiction == 0` → estatuas, dibujos, pósters
- quedarte con **cajas chicas** (alto < 15 % del alto de imagen) para no sesgar
  el modelo hacia retratos bien encuadrados

### Opcional — PeopleSansPeople (sintético Unity)
https://github.com/Unity-Technologies/PeopleSansPeople — ✅ Apache 2.0

Generás humanos con la cámara puesta a 4 m y ángulo cenital, oclusión densa y
poca luz. Etiquetas exhaustivas perfectas por construcción. Como refuerzo de
casos raros, nunca como set principal.

### ❌ No usar en el modelo que se despliega
CrowdHuman · WiderPerson · MOT17/MOT20 · VisDrone · CityPersons · LLVIP ·
SOMPT22 · IndoorCrowd (este último **prohíbe explícitamente uso en vigilancia**).
PANDA y TinyPerson: licencia no declarada en ningún lado → tratar como desconocida.

### Si igual usás CrowdHuman para prototipar: usá `vbox`, no `fbox`
El ODGT trae tres cajas por persona: `hbox` (cabeza), `vbox` (región **visible**)
y `fbox` (cuerpo completo **amodal**, extrapolado).

Usá **vbox**, por tres razones:
1. `fbox` destruye el recall en multitud vía NMS (IoU par >0,5 promedio de 2,4
   por imagen). Justo cuando más falta hace el detector, se cae.
2. `fbox` enseña a alucinar extensión: el objetivo de regresión incluye píxeles
   sin evidencia visual. Desestabiliza la localización y arruina los recortes
   que después consume tu cabeza `reid_cuerpo`.
3. **Consistencia**: D-Fire, Sohas y los sets de paquete usan todos extensión
   visible. Si solo la clase 2 llega con caja amodal, esa clase tiene una escala
   sistemáticamente distinta a las otras seis.

Filtrá además por visibilidad: descartá instancias con
`area(vbox)/area(fbox) < 0.25`, si no le enseñás que "persona" puede ser medio hombro.

**Y ojo con las regiones `ignore`** (`tag == "mask"` o `extra.ignore == 1`):
- Si las convertís en cajas de persona → alertas falsas sobre maniquíes y pósters.
- Si las borrás y dejás la imagen → hay decenas de personas reales sin etiqueta
  ahí adentro, y estás entrenando al modelo a **no detectar gente en multitudes**.
  Para un sistema de prevención eso es exactamente al revés.
- Correcto: pintarlas de gris antes de entrenar, o excluir esos anchors de la pérdida.

**Trampa que te va a hacer abortar el chequeo del dataset:** las cajas de
CrowdHuman pueden tener coordenadas negativas o mayores al tamaño de imagen
(los `fbox` amodales se salen del encuadre). Hay que clipear a los bordes antes
de escribir las etiquetas, o `entrenar_objetos_cuda.py` corta con
"coordenadas fuera de 0..1". Lo mismo aplica a COCO (`iscrowd==1` es ignore) y
VisDrone (`score==0` es ignore).

---

## 3 y 4 · PISTOLA y CUCHILLO

**Leé la sección "Expectativas realistas" al final antes de invertir semanas acá.**

### ★ Monash Guns Dataset (MGD) — la mejor relación licencia/realismo
https://github.com/MarcusLimJunYi/Monash-Guns-Dataset

- **5.500 imágenes** de 250 videos CCTV · Pascal VOC
- **Escenificado sobre cámaras CCTV reales**: actores con réplica, variando
  altura, ángulo y profundidad, día/tarde/noche, interior y exterior con clutter
- ✅ **MIT**
- Solo pistola, sin cuchillo ni celular. Pistola media: **25 px**

### ★ ACF — Armed CCTV Footage (Mahidol) — pistola + cuchillo, tu geometría exacta
Paper: https://www.mdpi.com/1424-8220/22/19/7158

- **8.319 frames** 1920×1080 (4.426 pistola + 3.559 cuchillo)
- Rig CCTV real Hikvision, **2 cámaras a 2,8 m de altura, 30-35° de inclinación**,
  3 estacionamientos + 1 pasillo. Es literalmente tu caso de instalación.
- Etiquetas de **49×62 px** promedio. Con **tiling 2×2 el mAP subió ×10,22**.
- ⚠️ **No encontré link de descarga público ni licencia declarada** — hay que
  escribirle a los autores (Hnoohom et al., Mahidol). Si te lo dan, es la pieza
  que más te aporta de toda la lista.

### Open Images V7 — cuchillo con licencia limpia
✅ anotaciones CC BY 4.0 · imágenes CC BY 2.0

| Clase | MID |
|---|---|
| Handgun | `/m/0gxl3` |
| Knife | `/m/04ctx` |
| Kitchen knife | `/m/058qzx` |
| Dagger | `/m/02gzp` |
| Rifle / Shotgun (negativos) | `/m/06c54` / `/m/06nrc` |

Son fotos de Flickr: objetos grandes, centrados, nítidos. Sirven para
pre-entrenar features, **no** para separar un objeto de 25 px de otro de 25 px.

### CCTV-Gun — para MEDIR, no para entrenar
https://github.com/srikarym/CCTV-Gun · paper https://arxiv.org/abs/2303.10703

✅ Apache-2.0 (código y anotaciones; las imágenes las bajás de cada fuente).
Es el único protocolo de evaluación honesto que existe: anota oclusión, blur y
**"objetos similares"** (celulares incluidos). Si no medís ahí, no sabés nada.

### OD-WeaponDetection / Sohas — solo por el pareado con celular
https://github.com/ari-dasci/OD-WeaponDetection · https://sci2s.ugr.es/weapons-detection

| Subconjunto | Imgs | Desglose |
|---|---|---|
| Sohas detection | 5.859 | knife 2.349 · pistol 1.663 · **smartphone 893** · purse 644 · bill 584 · card 313 |
| Pistol detection | 3.000 | 1 clase |
| Knife detection | 2.078 | 1 clase |

Descarga cómoda: https://datasetninja.com/od-weapon-detection-sohas-detection (1,73 GB)

⚠️ **CC BY-SA 4.0 con dos asteriscos:** (a) la UGR relicenció imágenes
scrapeadas de terceros que no le pertenecen, así que esa licencia es
jurídicamente hueca; (b) ShareAlike — si un juez considerara los pesos obra
derivada, tendrías que liberar el modelo. La lectura mayoritaria dice que no,
pero es zona gris.

**Su único valor real:** es la única fuente pública y descargable con la clase
**smartphone anotada junto a pistola y cuchillo en las mismas imágenes**.

### ❌ No usar
| Dataset | Motivo |
|---|---|
| USRT / Mock Attack (Sevilla) | CC BY-**NC** 4.0. Es el mejor dato de CCTV real que existe (5.149 frames, simulacro autorizado) — pedí permiso comercial por escrito, tarda semanas |
| Gun Action Recognition (Data in Brief 2024) | CC BY-NC 3.0. Sus 118 videos `No_Gun` son negativos duros excelentes; pedir permiso |
| UCF-Crime | **ninguna licencia declarada**, videos recopilados de noticias |
| YouTube-GDD | 20.934 objetos "large" contra 1.106 "small" — entrena el sesgo exactamente al revés |
| Roboflow `weapon-detection-n5y5l` | fotos de catálogo (`glock-pistol-images_56.jpg`), 1 clase que mezcla pistola y granada |
| Roboflow `weapon-detection-cctv-v3` | taxonomía rota: `pistol, gun, guns, weapon, Knife, knife...` duplicadas por mayúsculas |

---

## 5 · CELULAR

Vale la pena tenerla, y no solo como clase propia: **etiquetar celulares fuerza
al modelo a construir la frontera contra la pistola.** Si hay gente sosteniendo
celulares sin etiquetar, el modelo los ve como fondo y nunca recibe gradiente
para separarlos.

Está medido: en CCTV-Gun el AP cae de **50,92 a 45,05** solo por presencia de
objetos pequeños similares, con los celulares nombrados explícitamente.

**La condición que lo decide todo:** los celulares tienen que estar a la **misma
escala y geometría** que las pistolas. Agregar 6.000 celulares de COCO a 200 px
no enseña nada sobre separar un celular de 25 px de una pistola de 25 px.

| Fuente | Cantidad | Licencia |
|---|---|---|
| **Grabado por vos** | — | ✅ insustituible |
| Sohas smartphone | 893 cajas | ⚠️ CC BY-SA |
| Open Images `Mobile phone` `/m/050k8` | — | ✅ CC BY |
| COCO `cell phone` (id 67) | **6.422 instancias / 4.803 imgs** | ✅ anotaciones CC BY 4.0 |

❌ Los datasets de celular de Roboflow son de e-waste, PCB o distracción del
conductor (cámara a 50 cm). Geometría equivocada.

---

## 6 · PAQUETE

### ★ package-at-front-door — el único con perspectiva correcta
https://universe.roboflow.com/package-detection/package-at-front-door

- **1.293 imágenes** · 1 clase `package` · ✅ **MIT**
- Perspectiva de **cámara de timbre / puerta de entrada** — cámara fija exterior

### ★ boxes-iqjbg — volumen limpio
https://universe.roboflow.com/joel-tnzu2/boxes-iqjbg
- **3.064 imágenes** · 1 clase · ✅ CC BY 4.0

### Complementos
| Dataset | Imgs | Licencia |
|---|---|---|
| https://universe.roboflow.com/hamid-rezaie/cardboard_box_mask | 6.667 | ✅ CC BY 4.0 (segmentación → Roboflow exporta YOLO) |
| https://universe.roboflow.com/hprivs/packages-iepao | 1.503 | ✅ CC BY 4.0 |
| https://public.roboflow.com/object-detection/packages-dataset | 250 | ✅ **CC0** — poquísimo pero perspectiva perfecta, usalo de **val de dominio** |
| Open Images `Box` `/m/025dyy` | — | ✅ CC BY |
| LVIS `box` | 1.828 img / 7.855 cajas | ✅ CC BY 4.0 (imágenes = COCO) |

### Objeto abandonado (bolsos, valijas, mochilas)
https://universe.roboflow.com/sahanaworkspace/abandoned-object
- 2.729 imgs · 20 clases (Box, Suitcase, Backpack, Handbag, Briefcase…)
- Material CCTV real (los nombres de archivo lo delatan)
- ⚠️ **ODbL v1.0** — share-alike sobre la base de datos derivada, revisalo

### ❌ No usar
- **SKU-110K**: *"solely for academic and non-commercial purposes"*. Además son
  góndolas de supermercado, no paquetes.
- **Parcel3D**: no comercial.
- **PETS2006**: el dominio ya no resuelve DNS. Muerto.
- **AVSS2007 i-LIDS**: la página da 404; es propiedad del Home Office UK.
- **ABODA**: repo vivo pero **no trae bounding boxes**, solo 11 videos `.avi`.

---

## Lo que no está en ningún lado

- **Humo incipiente en interiores con cajas: no existe.** Nada. Nemo, FIgLib y
  Pyro-SDIS son todos de torres forestales a kilómetros. IFireSmoke y
  DetectiumFire son de fuego ya declarado. Esa brecha la llenás vos o no se llena.
- **Ni COCO, ni Open Images, ni LVIS, ni Objects365 tienen `smoke` ni `flame`.**
- **No hay ningún dataset público de robo de paquetes** con imágenes anotadas.

---

## Expectativas realistas por clase

Antes de repartir semanas de trabajo, esto es lo que dicen los números medidos:

**Fácil:** paquete, persona, celular (objetos grandes, rígidos, contrastados).

**Difícil — presupuestá que van a rendir claramente peor:** pistola y cuchillo.

Tamaños medios medidos en CCTV-Gun (arXiv 2303.10703):

| Dataset | Pistola | Persona |
|---|---|---|
| MGD (CCTV escenificado) | **25 px** | 158 px |
| USRT (simulacro real) | **47 px** | 319 px |
| UCF-Crime (crimen real) | **16 px** | 79 px |

Todo eso cae en "small" de COCO (<32×32), el régimen donde los detectores peor
rinden. Y la generalización cruzada es brutal: entrenar en MGD+USRT y evaluar en
crimen real da **10,3 AP**, y **3,7 AP** para handgun con algunos modelos.

Regla práctica derivada de la óptica (~20 cm de arma, 1080p, 60° HFOV):

| Píxeles sobre el arma | Distancia máx. @1080p | @4K |
|---|---|---|
| 60 px (cómodo) | ~6 m | ~12 m |
| 40 px (mínimo operativo) | **~9 m** | ~18 m |
| 25 px (marginal) | ~15 m | ~29 m |

**Tres cosas no negociables si vas a intentar la clase pistola:**
1. **Tiling 2×2 o 3×3 en inferencia.** ×10,22 de mejora medido en ACF. Sin esto
   no hay producto.
2. **N frames consecutivos** antes de alertar (el paper de Granada usaba 5).
3. **Verificación humana en el lazo.** Ningún fabricante del mercado despliega
   esto en lazo cerrado: Omnilert exige verificación humana obligatoria antes de
   emitir alerta; ZeroEyes habla de detecciones "confirmadas" — por humanos.
   Un paper de BRACIS 2025 (publicado feb-2026) concluye que las mejoras en
   detección vienen con *"un aumento notable de falsos positivos"*. Si tu
   producto promete alarma automática sin humano, se apaga la primera semana.

Esto encaja bien con tu arquitectura: para eso está el gate del VLM en
`06_fusion_decision`. La cabeza de objetos marca el candidato, el VLM verifica.

---

## Plan de mezcla concreto

```
CLASE 0 · humo
    D-Fire (deduplicado)                 ~5.900 imgs   CC0
    Pyro-SDIS (muestreado)                3.000 imgs   Apache-2.0
CLASE 1 · llama
    D-Fire (deduplicado)                 ~5.800 imgs   CC0
    MS-FSDB positivas                    ~3.600 imgs   MIT
CLASE 2 · persona
    metraje propio auto-etiquetado        2.000 imgs   propio  ★
    MEVA auto-etiquetado                  1.500 imgs   CC BY 4.0
    Open Images Person filtrado           1.500 imgs   CC BY
CLASE 3 · pistola
    Monash Guns (MGD)                     5.500 imgs   MIT
    grabado propio con réplica            1.500 imgs   propio  ★
CLASE 4 · cuchillo
    Open Images Knife+Kitchen knife       ~2.000 imgs  CC BY
    grabado propio                        1.500 imgs   propio  ★
CLASE 5 · celular
    Sohas smartphone                        893 cajas  CC BY-SA
    COCO cell phone                       4.800 imgs   CC BY (anot.)
    grabado propio (misma escala!)        1.000 imgs   propio  ★
CLASE 6 · paquete
    package-at-front-door                 1.293 imgs   MIT
    boxes-iqjbg                           3.064 imgs   CC BY 4.0
    Open Images Box                       ~2.000 imgs  CC BY

NEGATIVOS (labels vacíos, sin cajas)
    D-Fire background                     9.838 imgs   CC0
    FireAndSmoke clase "other"            ~3.000 imgs  CC BY 4.0
    SAINetset negativos                   5.351 imgs   CC BY 4.0
    frames vacíos de tus cámaras          1.000 imgs   propio  ★
```

Las 4 líneas marcadas ★ son las que más mueven la aguja y las únicas que nadie
te puede dar. Todo lo demás es relleno de diversidad.

**Regla de oro del remapeo:** cada dataset numera desde 0. D-Fire es el único
que ya coincide con tu numeración (0=humo, 1=llama). Todo el resto hay que
remapearlo. `entrenar_objetos_cuda.py` aborta si encuentra índices fuera de 0..6
o coordenadas fuera de 0..1 — dejalo que revise antes de cada corrida.

**Deduplicá el pool COMPLETO con pHash, no dataset por dataset.** Varias de
estas fuentes se reciclan entre sí (FASDD declara reutilizar datasets abiertos,
y varios proyectos de Roboflow son reempaquetados de D-Fire).

**Bajá los datasets de Roboflow en la versión SIN augmentation.** Por defecto
exportan con augmentation 3x aplicada: no es más dato, es la misma imagen
triplicada, y le miente a cualquier chequeo de balance.

---

## Pendientes de verificar (no confirmado en esta investigación)

- **Licencia de FASDD** — crítico si lo querés usar. El CC BY 4.0 visible es del paper.
- **Link de descarga y licencia de ACF** (Mahidol) — MDPI bloqueó el PDF completo.
- **Licencia de PANDA y de WiderPerson** — no declaradas en fuente primaria.
- **Conteos por clase de Open Images V7 y Objects365** — nadie los publica.
- **Objects365** tiene `Gun` con volumen, pero sus términos oficiales dicen que
  las imágenes son *"for academic purpose only"* → no autorizado para producto.

## Nota buena

El aviso de **AGPL-3.0 de Ultralytics** (que obliga a liberar el servicio o
comprar licencia enterprise) **no te aplica**: tu `objects_head.py` está escrito
sobre torchvision (BSD-3), no sobre Ultralytics. Es una bala esquivada sin
querer, y conviene no perder esa posición.
