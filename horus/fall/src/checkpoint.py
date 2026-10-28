import os
import subprocess
import time

import torch

# Un checkpoint no es solo los pesos: sin saber con que configuracion, con que
# datos y con que version del codigo salio, dos archivos .pt son
# indistinguibles. Aca se guarda todo junto y se lee igual los archivos viejos
# (state_dict crudo) que los nuevos.

CLAVE_FORMATO = "_horus_checkpoint"


def hash_git():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    text=True, stderr=subprocess.DEVNULL).strip()
        sucio = subprocess.check_output(["git", "status", "--porcelain"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return h + ("+sucio" if sucio else "")
    except Exception:
        return "?"


def _plano(v):
    """Los metadatos van en tipos nativos: un float64 de numpy hace que el
    checkpoint no se pueda leer con weights_only=True, que es el modo seguro."""
    if isinstance(v, dict):
        return {str(k): _plano(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plano(x) for x in v]
    if hasattr(v, "item") and getattr(v, "ndim", 0) == 0:
        v = v.item()
    # tipo EXACTO: TorchVersion y np.str_ son subclases de str y se guardarian
    # con su clase propia, que weights_only=True tampoco acepta
    if type(v) in (str, int, float, bool) or v is None:
        return v
    if isinstance(v, (str, int, float, bool)):
        return type(v).__mro__[-2](v)
    return str(v)


def guardar(ruta, modelo, config=None, metricas=None, extra=None, miembros=None):
    """`miembros`: varios modelos de la misma receta (un ensamble de semillas)
    en UN archivo. Asi el despliegue sigue siendo un solo .pt: no cambian el
    lanzador, el manifiesto de hashes ni el instalador. `state_dict` queda con
    el primero, y un lector que no conozca `state_dicts` carga un miembro
    valido en vez de romperse."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    paquete = {
        CLAVE_FORMATO: 1,
        "state_dict": modelo.state_dict(),
        "config": _plano(dict(config or {})),
        "metricas": _plano(dict(metricas or {})),
        "git": hash_git(),
        "fecha": time.strftime("%Y-%m-%d %H:%M"),
        "torch": str(torch.__version__),
    }
    paquete.update(_plano(extra or {}))
    if miembros:
        # despues de _plano a proposito: aplanaria los tensores a texto
        paquete["state_dicts"] = [m.state_dict() for m in miembros]
    torch.save(paquete, ruta)
    return ruta


def cargar(ruta, modelo=None, device="cpu"):
    """Devuelve (state_dict, metadatos). Si le paso un modelo, le carga los pesos.
    Acepta los checkpoints viejos, que son un state_dict pelado sin metadatos."""
    crudo = torch.load(ruta, map_location=device, weights_only=True)

    if isinstance(crudo, dict) and CLAVE_FORMATO in crudo:
        state_dict = crudo["state_dict"]
        meta = {k: v for k, v in crudo.items() if k != "state_dict"}
    else:
        state_dict = crudo
        meta = {"formato": "viejo (state_dict sin metadatos)"}

    if modelo is not None:
        modelo.load_state_dict(state_dict)
    return state_dict, meta


def guardar_umbral(ruta, umbral, config=None, metricas=None):
    """Escribe el umbral calibrado en los metadatos de un checkpoint ya
    guardado, sin tocar nada mas. Lo usa 21_calibrar_umbral.py.

    No re-guarda con `guardar`: eso perderia los miembros de un ensamble y
    pisaria `git` y `fecha`, que tienen que seguir siendo los del
    entrenamiento, no los de la calibracion."""
    paquete = torch.load(ruta, map_location="cpu", weights_only=True)
    if not (isinstance(paquete, dict) and CLAVE_FORMATO in paquete):
        # un state_dict pelado no tiene donde guardar el umbral: se envuelve
        paquete = {CLAVE_FORMATO: 1, "state_dict": paquete, "config": {}, "metricas": {},
                   "git": "?", "fecha": "?", "torch": str(torch.__version__)}
    paquete.setdefault("config", {}).update(_plano(dict(config or {})))
    paquete["config"]["umbral"] = float(umbral)
    paquete.setdefault("metricas", {}).update(_plano(dict(metricas or {})))
    torch.save(paquete, ruta)


def aplicar_config(meta, cfg):
    """Pasa a la ConfigMaquina lo que el checkpoint traiga guardado. Hoy el
    umbral: es una propiedad del modelo (cada entrenamiento calibra distinto),
    no de la configuracion, asi que viaja con los pesos. Ver 21_calibrar_umbral.py."""
    guardado = (meta or {}).get("config", {}).get("umbral")
    if guardado is not None:
        cfg.umbral = float(guardado)
        return True
    return False


def describir(ruta):
    """Una linea legible con lo que hay adentro de un checkpoint."""
    try:
        _, meta = cargar(ruta)
    except Exception as e:
        # no caigo al modo inseguro (weights_only=False) automaticamente: cargar
        # pickle arbitrario es justo lo que el modo seguro evita. Aviso y sigo.
        return f"{os.path.basename(ruta)}: ILEGIBLE en modo seguro ({type(e).__name__}) — regeneralo"
    if "formato" in meta:
        return f"{os.path.basename(ruta)}: {meta['formato']}"
    m = meta.get("metricas", {})
    resumen = ", ".join(f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
                        for k, v in m.items()) or "sin metricas"
    return (f"{os.path.basename(ruta)}: git={meta.get('git')} fecha={meta.get('fecha')} | {resumen}")


if __name__ == "__main__":
    import glob
    import sys
    patrones = sys.argv[1:] or ["../checkpoints/*.pt", "../checkpoints/loso/*.pt"]
    for patron in patrones:
        for r in sorted(glob.glob(patron)):
            print(describir(r))
