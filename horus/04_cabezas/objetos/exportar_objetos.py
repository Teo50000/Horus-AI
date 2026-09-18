# -*- coding: utf-8 -*-
"""
exportar_objetos.py · HORUS — export de la cabeza de objetos a ONNX y TensorRT.

La arquitectura (04_cabezas/LEEME.txt) dice "ResNet-50+FPN, TensorRT INT8".
Este script es el que produce ese engine.

Qué se exporta
--------------
Solo la parte estática: preproceso normalizado -> backbone -> FPN -> torres ->
(logits, deltas, embedding). El NMS y el decode se quedan en PyTorch dentro de
`objects_engine.ObjectsEngine._postproceso`, porque:
  - ya están vectorizados en GPU y cuestan microsegundos,
  - meter NMS adentro del engine ata el umbral y el topk al build (cambiar el
    umbral obligaría a reconstruir el .plan, y en vigilancia eso se toca),
  - el plugin EfficientNMS_TRT no está en todas las instalaciones.

Flujo típico
------------
    # 1) ONNX
    python exportar_objetos.py --pesos checkpoints/head_best.pt --onnx

    # 2) engine FP16 (rápido de construir, ~2x sobre PyTorch fp16)
    python exportar_objetos.py --pesos checkpoints/head_best.pt --trt --fp16

    # 3) engine INT8 (lo que pide la arquitectura; necesita imágenes reales)
    python exportar_objetos.py --pesos checkpoints/head_best.pt --trt --int8 \\
        --calib-dir datasets/mezcla_v1/images/train --calib-num 500

    # 4) usarlo
    python objects_engine.py 0 --ver --trt engines/objetos_int8.plan

Verificación (compara TRT contra PyTorch sobre imágenes reales o ruido):
    python exportar_objetos.py --verificar engines/objetos_fp16.plan \\
        --pesos checkpoints/head_best.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_AQUI = os.path.dirname(os.path.abspath(__file__))
if _AQUI not in sys.path:
    sys.path.insert(0, _AQUI)

from objects_engine import (  # noqa: E402
    EngineConfig,
    ObjectsEngine,
)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- #
# 1) ONNX
# --------------------------------------------------------------------------- #
def exportar_onnx(
    cfg: EngineConfig,
    salida: Path,
    opset: int = 18,
    batch_dinamico: bool = True,
    simplificar: bool = True,
) -> Path:
    """Exporta FusedObjectsNet a ONNX. Siempre en fp32: la precisión reducida
    la decide TensorRT en el build, no el ONNX."""
    cfg_exp = EngineConfig(**{**cfg.__dict__})
    cfg_exp.precision = "fp32"
    cfg_exp.cuda_graphs = False
    cfg_exp.compile = False
    cfg_exp.channels_last = False
    cfg_exp.trt_engine = None
    cfg_exp.warmup = 0

    if not cfg_exp.pesos:
        print("[onnx] ⚠ estás exportando SIN pesos entrenados (--pesos). El "
              "engine va a salir con ruido. Si es a propósito (medir "
              "velocidad) seguí; si no, cancelá y pasá el checkpoint.")

    eng = ObjectsEngine(cfg_exp)
    modelo = eng.model.float().eval().to("cpu")

    s = cfg.input_size
    dummy = torch.zeros(1, 3, s, s)

    salida.parent.mkdir(parents=True, exist_ok=True)
    dyn = {"imagen": {0: "batch"},
           "logits": {0: "batch"},
           "deltas": {0: "batch"},
           "embedding": {0: "batch"}} if batch_dinamico else None

    print(f"[onnx] exportando (opset {opset}, input {s}x{s}, "
          f"batch {'dinámico' if batch_dinamico else 'fijo 1'})...")

    import inspect
    kw = dict(input_names=["imagen"],
              output_names=["logits", "deltas", "embedding"],
              dynamic_axes=dyn, opset_version=opset, do_constant_folding=True)
    # torch >= 2.6 guarda los pesos en un .onnx.data aparte. TensorRT parseando
    # desde bytes no lo encuentra, así que forzamos todo en un solo archivo.
    if "external_data" in inspect.signature(torch.onnx.export).parameters:
        kw["external_data"] = False

    with torch.no_grad():
        torch.onnx.export(modelo, dummy, str(salida), **kw)

    _inlinear_pesos(salida)

    if simplificar:
        try:
            import onnx
            from onnxsim import simplify
            m = onnx.load(str(salida))
            m_s, ok = simplify(m)
            if ok:
                onnx.save(m_s, str(salida))
                print("[onnx] simplificado con onnx-simplifier")
            else:
                print("[onnx] onnxsim no validó el modelo; dejo el original")
        except ImportError:
            print("[onnx] onnxsim no instalado (pip install onnxsim) — se saltea")
        except Exception as e:
            print(f"[onnx] onnxsim falló ({e}); dejo el original")

    print(f"[onnx] listo: {salida}  ({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


def _inlinear_pesos(salida: Path) -> None:
    """Mete los pesos dentro del .onnx y borra el .onnx.data suelto: un archivo
    autocontenido es lo único que TensorRT parsea sin sorpresas."""
    extra = Path(str(salida) + ".data")
    try:
        import onnx
        m = onnx.load(str(salida))          # resuelve el external data
        onnx.save(m, str(salida), save_as_external_data=False)
        if extra.exists():
            extra.unlink()
            print("[onnx] pesos incorporados al .onnx (borré el .onnx.data)")
    except ImportError:
        if extra.exists():
            print("[onnx] ⚠ quedó un .onnx.data aparte y no tenés el paquete "
                  "'onnx' para incorporarlo. Instalá: pip install onnx")
    except Exception as e:
        print(f"[onnx] no pude incorporar los pesos ({e}); dejo el .onnx.data")


# --------------------------------------------------------------------------- #
# 2) Calibrador INT8
# --------------------------------------------------------------------------- #
def _preprocesar_para_calib(ruta: Path, size: int) -> np.ndarray:
    """MISMO preproceso que el motor en producción. Si acá difiere, la
    calibración INT8 sale sesgada y el modelo pierde precisión."""
    import cv2
    img = cv2.imread(str(ruta))
    if img is None:
        raise RuntimeError(f"no pude leer {ruta}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _hacer_calibrador(trt, calib_dir: Path, size: int, batch: int,
                      n_max: int, cache: Path):
    """Entropy calibrator v2 — el que mejor anda para detección."""

    class Calibrador(trt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            super().__init__()
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            self.archivos = sorted(
                p for p in calib_dir.iterdir() if p.suffix.lower() in exts
            )[:n_max]
            if not self.archivos:
                raise SystemExit(
                    f"No hay imágenes en {calib_dir}. INT8 necesita fotos REALES "
                    "de tus cámaras (300-1000 alcanzan). Ver datasets/LEEME.txt."
                )
            self.batch = batch
            self.i = 0
            self.cache = cache
            self.buf = torch.empty((batch, 3, size, size), device="cuda",
                                   dtype=torch.float32)
            print(f"[int8] calibrando con {len(self.archivos)} imágenes de "
                  f"{calib_dir}")

        def get_batch_size(self) -> int:
            return self.batch

        def get_batch(self, names, p_str=None):
            if self.i + self.batch > len(self.archivos):
                return None
            lote = self.archivos[self.i:self.i + self.batch]
            arr = np.stack([_preprocesar_para_calib(p, self.buf.shape[-1])
                            for p in lote])
            self.buf.copy_(torch.from_numpy(arr).cuda(), non_blocking=True)
            self.i += self.batch
            if self.i % (self.batch * 20) == 0:
                print(f"[int8]   {self.i}/{len(self.archivos)}")
            return [int(self.buf.data_ptr())]

        def read_calibration_cache(self):
            if self.cache.exists():
                print(f"[int8] usando cache {self.cache}")
                return self.cache.read_bytes()
            return None

        def write_calibration_cache(self, cache_bytes):
            self.cache.write_bytes(cache_bytes)
            print(f"[int8] cache guardada en {self.cache}")

    return Calibrador()


# --------------------------------------------------------------------------- #
# 3) Build del engine TensorRT
# --------------------------------------------------------------------------- #
def construir_trt(
    onnx_path: Path,
    salida: Path,
    fp16: bool = True,
    int8: bool = False,
    calib_dir: Optional[Path] = None,
    calib_num: int = 500,
    input_size: int = 384,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 4,
    workspace_gb: float = 4.0,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError:
        raise SystemExit(
            "TensorRT no está instalado.\n"
            "  pip install tensorrt        (o el paquete de NVIDIA para tu CUDA)\n"
            "Alternativa sin instalar nada: usar el motor PyTorch con\n"
            "  --precision fp16 --cuda-graphs, que ya da la mayor parte de la ganancia."
        )

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)

    parser = trt.OnnxParser(network, logger)
    print(f"[trt] parseando {onnx_path}")
    # parse_from_file resuelve rutas relativas (external data); parse(bytes) no.
    if hasattr(parser, "parse_from_file"):
        ok = parser.parse_from_file(str(onnx_path))
    else:
        ok = parser.parse(onnx_path.read_bytes())
    if not ok:
        for i in range(parser.num_errors):
            print("  ", parser.get_error(i))
        raise SystemExit("[trt] el ONNX no parseó")

    config = builder.create_builder_config()
    ws = int(workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):        # TRT >= 8.4
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    else:                                                # TRT viejo
        config.max_workspace_size = ws

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("[trt] ⚠ esta GPU no tiene FP16 rápido")
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] FP16 activado")

    calibrador = None
    if int8:
        if not builder.platform_has_fast_int8:
            print("[trt] ⚠ esta GPU no tiene INT8 rápido")
        if calib_dir is None:
            raise SystemExit("[trt] INT8 necesita --calib-dir con imágenes reales")
        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 + INT8 juntos: TRT elige por capa la que más conviene.
        config.set_flag(trt.BuilderFlag.FP16)
        cache = salida.with_suffix(".calib")
        calibrador = _hacer_calibrador(trt, calib_dir, input_size,
                                       max(1, min(8, max_batch)), calib_num, cache)
        config.int8_calibrator = calibrador
        print("[trt] INT8 activado (entropy calibrator v2)")

    profile = builder.create_optimization_profile()
    s = input_size
    profile.set_shape("imagen",
                      (min_batch, 3, s, s), (opt_batch, 3, s, s), (max_batch, 3, s, s))
    config.add_optimization_profile(profile)
    if int8 and hasattr(config, "set_calibration_profile"):
        config.set_calibration_profile(profile)

    salida.parent.mkdir(parents=True, exist_ok=True)
    print("[trt] construyendo engine (esto tarda: FP16 minutos, INT8 más)...")
    t0 = time.perf_counter()
    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise SystemExit("[trt] el build falló")
        salida.write_bytes(bytes(plan))
    else:                                                # TRT muy viejo
        engine = builder.build_engine(network, config)
        if engine is None:
            raise SystemExit("[trt] el build falló")
        salida.write_bytes(engine.serialize())

    dt = time.perf_counter() - t0
    print(f"[trt] listo en {dt/60:.1f} min -> {salida} "
          f"({salida.stat().st_size / 1e6:.1f} MB)")
    return salida


# --------------------------------------------------------------------------- #
# 4) Runner: usar el .plan desde el motor
# --------------------------------------------------------------------------- #
class TRTRunner:
    """Ejecuta un engine TensorRT con tensores de PyTorch (sin pycuda).

    Devuelve (logits, deltas, embedding) como tensores CUDA, listos para que
    `ObjectsEngine._postproceso` haga decode + NMS igual que con PyTorch.
    """

    def __init__(self, ruta: str, device: str = "cuda") -> None:
        import tensorrt as trt

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(Path(ruta).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"no pude deserializar {ruta}")
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.api_v3 = hasattr(self.ctx, "set_tensor_address")

        if self.api_v3:                                   # TRT >= 8.5
            self.nombres = [self.engine.get_tensor_name(i)
                            for i in range(self.engine.num_io_tensors)]
            self.entradas = [n for n in self.nombres
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.salidas = [n for n in self.nombres if n not in self.entradas]
        else:                                             # TRT 8.0-8.4
            self.nombres = [self.engine.get_binding_name(i)
                            for i in range(self.engine.num_bindings)]
            self.entradas = [n for n in self.nombres
                             if self.engine.binding_is_input(n)]
            self.salidas = [n for n in self.nombres if n not in self.entradas]

        self._buffers: Dict[Tuple[str, int], torch.Tensor] = {}

    def _dtype_torch(self, trt_dtype) -> torch.dtype:
        trt = self.trt
        m = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.INT32: torch.int32,
            trt.DataType.INT8: torch.int8,
            trt.DataType.BOOL: torch.bool,
        }
        for nombre, td in (("BF16", torch.bfloat16), ("INT64", torch.int64)):
            if hasattr(trt.DataType, nombre):
                m[getattr(trt.DataType, nombre)] = td
        return m.get(trt_dtype, torch.float32)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.to(self.device, dtype=torch.float32).contiguous()
        b = x.shape[0]
        nombre_in = self.entradas[0]

        if self.api_v3:
            self.ctx.set_input_shape(nombre_in, tuple(x.shape))
            self.ctx.set_tensor_address(nombre_in, int(x.data_ptr()))
            outs = []
            for n in self.salidas:
                shape = tuple(self.ctx.get_tensor_shape(n))
                dt = self._dtype_torch(self.engine.get_tensor_dtype(n))
                key = (n, b)
                buf = self._buffers.get(key)
                if buf is None or tuple(buf.shape) != shape:
                    buf = torch.empty(shape, device=self.device, dtype=dt)
                    self._buffers[key] = buf
                self.ctx.set_tensor_address(n, int(buf.data_ptr()))
                outs.append(buf)
            stream = torch.cuda.current_stream(self.device)
            if not self.ctx.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("execute_async_v3 falló")
        else:
            idx_in = self.engine.get_binding_index(nombre_in)
            self.ctx.set_binding_shape(idx_in, tuple(x.shape))
            bindings = [0] * self.engine.num_bindings
            bindings[idx_in] = int(x.data_ptr())
            outs = []
            for n in self.salidas:
                i = self.engine.get_binding_index(n)
                shape = tuple(self.ctx.get_binding_shape(i))
                key = (n, b)
                buf = self._buffers.get(key)
                if buf is None or tuple(buf.shape) != shape:
                    buf = torch.empty(shape, device=self.device,
                                      dtype=torch.float32)
                    self._buffers[key] = buf
                bindings[i] = int(buf.data_ptr())
                outs.append(buf)
            stream = torch.cuda.current_stream(self.device)
            if not self.ctx.execute_async_v2(bindings, stream.cuda_stream):
                raise RuntimeError("execute_async_v2 falló")

        por_nombre = dict(zip(self.salidas, outs))
        logits = por_nombre.get("logits", outs[0]).float()
        deltas = por_nombre.get("deltas", outs[1]).float()
        emb = por_nombre.get("embedding", outs[-1]).float()
        return logits, deltas, emb


# --------------------------------------------------------------------------- #
# 5) Verificación TRT vs PyTorch
# --------------------------------------------------------------------------- #
def verificar(plan: Path, cfg: EngineConfig, imagenes: Optional[Path],
              n: int = 8) -> int:
    print("=" * 70)
    print(f"VERIFICACIÓN · {plan.name} vs PyTorch fp32")
    print("=" * 70)

    cfg_ref = EngineConfig(**{**cfg.__dict__})
    cfg_ref.precision = "fp32"
    cfg_ref.cuda_graphs = False
    cfg_ref.trt_engine = None
    ref = ObjectsEngine(cfg_ref)

    cfg_trt = EngineConfig(**{**cfg.__dict__})
    cfg_trt.trt_engine = str(plan)
    mot = ObjectsEngine(cfg_trt)

    s = cfg.input_size
    frames: List[np.ndarray] = []
    if imagenes and imagenes.is_dir():
        import cv2
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for p in sorted(x for x in imagenes.iterdir()
                        if x.suffix.lower() in exts)[:n]:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(cv2.resize(img, (s, s)))
    if not frames:
        print("[verif] sin imágenes reales: uso ruido (los números van a ser "
              "pesimistas, el ruido no se parece a una escena)")
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (s, s, 3), dtype=np.uint8) for _ in range(n)]

    peor_logit, peor_iou, total, coinciden = 0.0, 1.0, 0, 0
    for f in frames:
        with torch.no_grad():
            x, _ = ref.pre([f])
            l_ref, d_ref, _ = ref.model(x)
            l_trt, d_trt, _ = mot.trt(x)
        peor_logit = max(peor_logit, float((l_ref - l_trt).abs().max()))

        r_ref = ref.infer(f, camera_id="v")
        r_trt = mot.infer(f, camera_id="v")
        total += len(r_ref.detections)
        for a in r_ref.detections:
            mejor = 0.0
            for b in r_trt.detections:
                if b.class_id != a.class_id:
                    continue
                ax1, ay1, ax2, ay2 = a.bbox_xyxy
                bx1, by1, bx2, by2 = b.bbox_xyxy
                ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                iy = max(0.0, min(ay2, by2) - max(ay1, by1))
                inter = ix * iy
                u = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
                mejor = max(mejor, inter / u if u > 0 else 0.0)
            if mejor >= 0.9:
                coinciden += 1
            peor_iou = min(peor_iou, mejor)

    print(f"  máx |Δlogit| PyTorch vs TRT : {peor_logit:.4f}")
    if total:
        print(f"  detecciones que reaparecen  : {coinciden}/{total} "
              f"({100*coinciden/total:.1f}%) con IoU>=0.9")
        print(f"  peor IoU                    : {peor_iou:.3f}")
    else:
        print("  (0 detecciones de referencia: probá con --umbral bajo o pesos "
              "entrenados)")

    ok = peor_logit < 0.5 and (total == 0 or coinciden >= 0.9 * total)
    print("\n" + ("El engine reproduce PyTorch ✅" if ok else
                  "Hay desvío: revisá calibración INT8 o probá FP16 ⚠️"))
    return 0 if ok else 1



# --------------------------------------------------------------------------- #
# Adelgazar el checkpoint para distribuirlo
# --------------------------------------------------------------------------- #
def adelgazar(entrada: Path, salida: Optional[Path] = None) -> Path:
    """Deja solo lo que la inferencia necesita.

    Un checkpoint de entrenamiento carga tres cosas que en producción sobran:
      - el estado de AdamW (dos momentos por parámetro: el doble del modelo),
      - los pesos crudos, si ya guardamos los EMA que son los que se usan,
      - scheduler y scaler.
    Sacarlos baja el archivo a un tercio y lo deja por debajo del límite de
    GitHub sin necesidad de LFS.
    """
    ck = torch.load(entrada, map_location="cpu")
    if not isinstance(ck, dict) or "head" not in ck:
        print(f"[slim] {entrada.name} ya es un state_dict pelado, no hay qué sacar")
        return entrada

    # Todo lo que el motor lee de un checkpoint, y nada más.
    #
    # 18/09: `backbone_tensores`, `backbone_huella` y `backbone_parcial` NO
    # estaban en esta lista. Esta función es de agosto; `incrustar_backbone()`
    # es de ayer, y no se conocían. Adelgazar un checkpoint autocontenido le
    # sacaba los 20 tensores irreconstruibles y lo dejaba dependiendo otra vez
    # de un `backbone.pt` al lado — exactamente el agujero que costó el v2.
    # Son 13,6 MB contra los 39,6 del optimizador: no es lo que engorda el
    # archivo, y es lo único que no se puede volver a generar.
    quedan = ("head", "clases", "fpn_levels", "anchor_sizes", "tam",
              "backbone_file", "backbone_pretrained", "metricas", "epoca",
              "backbone_tensores", "backbone_huella", "backbone_parcial")
    flaco = {k: ck[k] for k in quedan if k in ck}

    salida = salida or entrada.with_name(entrada.stem + "_deploy.pt")
    torch.save(flaco, salida)

    antes = entrada.stat().st_size / 1e6
    despues = salida.stat().st_size / 1e6
    sacados = [k for k in ck if k not in flaco]
    print(f"[slim] {entrada.name}: {antes:.0f} MB -> {salida.name}: "
          f"{despues:.0f} MB  ({100*(1-despues/antes):.0f}% menos)")
    print(f"[slim] sacado: {', '.join(sacados)}")
    if despues > 100:
        print("[slim] ⚠ sigue por encima de los 100 MB que GitHub rechaza. "
              "Usá Releases o Git LFS.")
    elif despues > 50:
        print("[slim] ⚠ GitHub avisa por encima de 50 MB (lo acepta igual).")
    else:
        print("[slim] entra en git sin LFS.")
    if "backbone_tensores" in flaco:
        print(f"[slim] los {len(flaco['backbone_tensores'])} tensores "
              f"irreconstruibles siguen adentro · huella "
              f"{flaco.get('backbone_huella')}")
    elif "backbone_huella" in ck:
        print("[slim] ⚠ el checkpoint de entrada tenía huella pero no tensores: "
              "el resultado NECESITA un backbone.pt al lado.")
    print("[slim] OJO: este archivo YA NO SIRVE para --reanudar. Guardá el "
          "original si pensás seguir entrenando.")
    return salida


# --------------------------------------------------------------------------- #
def incrustar_backbone(ruta_ck: Path, ruta_bb: Optional[Path],
                       salida: Optional[Path]) -> int:
    """Mete el backbone irreconstruible ADENTRO del checkpoint de la cabeza.

    El backbone son 27 M de parámetros, pero 23,5 M son el ResNet-50 de
    ImageNet, que se restaura en cualquier máquina. Los que no se pueden
    reconstruir —FPN, embed_head, GRU— son ~3,5 M (~14 MB) y entran de sobra
    adentro del `.pt` de la cabeza.

    Ahí es donde tendrían que haber estado siempre. `objetos_v2.pt` apuntaba a
    un `backbone.pt` al lado, ese archivo se perdió, y con él se perdió un
    modelo de 41 épocas que por lo demás está intacto. Un checkpoint que
    depende de un archivo suelto es un checkpoint a medias.
    """
    from shared_backbone import huella_backbone, pesos_no_imagenet

    if not ruta_ck.exists():
        print(f"No existe {ruta_ck}")
        return 1
    ck = torch.load(ruta_ck, map_location="cpu", weights_only=False)

    if ruta_bb is None:
        nombre = ck.get("backbone_file", "backbone.pt")
        ruta_bb = ruta_ck.parent / nombre
    if not ruta_bb.exists():
        print(f"No existe el backbone {ruta_bb}.\n"
              f"El checkpoint anota '{ck.get('backbone_file')}'. Sin ese "
              f"archivo exacto no hay nada que incrustar: el FPN no se puede "
              f"reconstruir ni sembrando la misma semilla.")
        return 1

    sd_bb = torch.load(ruta_bb, map_location="cpu", weights_only=False)
    no_in = pesos_no_imagenet(sd_bb)
    huella = huella_backbone(sd_bb)

    # Si el checkpoint ya traía huella, esto verifica que sea ESTE backbone.
    previa = ck.get("backbone_huella")
    if previa and previa != huella:
        print(f"El backbone no es el de este checkpoint.\n"
              f"  el checkpoint espera: {previa}\n"
              f"  {ruta_bb.name} tiene: {huella}\n"
              f"Incrustarlo dejaría un .pt que carga sin protestar y detecta "
              f"mal para siempre.")
        return 1

    ck["backbone_tensores"] = {k: v.detach().to(torch.float32).clone()
                               for k, v in no_in.items()}
    ck["backbone_parcial"] = True
    ck["backbone_huella"] = huella
    ck.setdefault("backbone_file", ruta_bb.name)

    salida = salida or ruta_ck.with_name(ruta_ck.stem + "_solo" + ruta_ck.suffix)
    torch.save(ck, salida)
    mb_extra = sum(v.numel() * 4 for v in no_in.values()) / 1e6
    print(f"[ok] {salida.name}  ({salida.stat().st_size/1e6:.1f} MB, "
          f"+{mb_extra:.1f} MB de backbone)")
    print(f"[ok] {len(no_in)} tensores irreconstruibles incrustados · "
          f"huella {huella}")
    print(f"[ok] a partir de acá el .pt no depende de ningún archivo al lado.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export ONNX / TensorRT de la cabeza de objetos")
    ap.add_argument("--pesos", default=None, help="state_dict de la cabeza (.pt)")
    ap.add_argument("--pesos-backbone", default=None)
    ap.add_argument("--input-size", type=int, default=384)
    ap.add_argument("--salida-dir", default="engines")

    ap.add_argument("--onnx", action="store_true", help="exportar ONNX")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--sin-simplificar", action="store_true")

    ap.add_argument("--trt", action="store_true", help="construir engine TensorRT")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default=None,
                    help="carpeta de imágenes reales para calibrar INT8")
    ap.add_argument("--calib-num", type=int, default=500)
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--workspace", type=float, default=4.0, help="GB")

    ap.add_argument("--adelgazar", default=None, metavar="CHECKPOINT",
                    help="dejar solo lo que necesita la inferencia (para subirlo "
                         "a git sin LFS)")
    ap.add_argument("--verificar", default=None, help="ruta a un .plan a validar")
    ap.add_argument("--verif-imgs", default=None)
    ap.add_argument("--umbral", type=float, default=0.30)
    ap.add_argument("--incrustar-backbone", dest="incrustar_backbone",
                    metavar="CHECKPOINT",
                    help="mete los tensores irreconstruibles del backbone "
                         "ADENTRO del checkpoint de la cabeza, y le pone huella. "
                         "A partir de ahí el .pt viaja solo.")
    ap.add_argument("--backbone", help="backbone.pt a incrustar (por defecto, "
                                       "el que anota el checkpoint)")
    ap.add_argument("--salida", help="dónde escribir el checkpoint incrustado "
                                     "(por defecto, al lado con sufijo _solo)")

    args = ap.parse_args()

    if args.incrustar_backbone:
        return incrustar_backbone(Path(args.incrustar_backbone),
                                  Path(args.backbone) if args.backbone else None,
                                  Path(args.salida) if args.salida else None)

    cfg = EngineConfig(
        pesos=args.pesos, pesos_backbone=args.pesos_backbone,
        input_size=args.input_size, max_batch=args.max_batch,
        score_thresh=args.umbral, warmup=2, cuda_graphs=False,
    )
    out_dir = Path(args.salida_dir)
    if not out_dir.is_absolute():
        out_dir = Path(_AQUI) / out_dir

    if args.adelgazar:
        adelgazar(Path(args.adelgazar))
        return 0

    if args.verificar:
        return verificar(Path(args.verificar), cfg,
                         Path(args.verif_imgs) if args.verif_imgs else None)

    if not (args.onnx or args.trt):
        ap.error("elegí --onnx y/o --trt (o --verificar)")

    onnx_path = out_dir / f"objetos_{args.input_size}.onnx"
    if args.onnx or args.trt:
        if onnx_path.exists() and not args.onnx:
            print(f"[onnx] reuso {onnx_path}")
        else:
            exportar_onnx(cfg, onnx_path, opset=args.opset,
                          simplificar=not args.sin_simplificar)

    if args.trt:
        if not (args.fp16 or args.int8):
            print("[trt] ni --fp16 ni --int8: construyo FP32 (lento, solo para comparar)")
        sufijo = "int8" if args.int8 else ("fp16" if args.fp16 else "fp32")
        plan = out_dir / f"objetos_{sufijo}.plan"
        construir_trt(
            onnx_path, plan, fp16=args.fp16 or args.int8, int8=args.int8,
            calib_dir=Path(args.calib_dir) if args.calib_dir else None,
            calib_num=args.calib_num, input_size=args.input_size,
            min_batch=args.min_batch, opt_batch=args.opt_batch,
            max_batch=args.max_batch, workspace_gb=args.workspace,
        )
        print(f"\nUsalo con:\n  python objects_engine.py 0 --ver --trt {plan}")
        print(f"Validalo con:\n  python exportar_objetos.py --verificar {plan} "
              f"--pesos {args.pesos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
