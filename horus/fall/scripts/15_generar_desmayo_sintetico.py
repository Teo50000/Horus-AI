import numpy as np
import os

CARPETA = "../data/processed"
ENTRADA = os.path.join(CARPETA, "todos.npy")
SALIDA = os.path.join(CARPETA, "todos_con_desmayo.npy")

# factores de estiramiento temporal: cuántas veces más lenta queda la caída.
# varios factores en vez de uno solo para no meter todas las muestras sintéticas
# con la misma "velocidad" -- si no, el modelo podría terminar aprendiendo
# ese único ritmo artificial en vez del patrón general de colapso lento.
FACTORES_ESTIRAMIENTO = [1.5, 2.0, 3.0]


def es_caida_transicion(item):
    """Solo estiro la transición real (de pie -> piso), no 'fallen' (ya en el piso,
    estático) ni otras subclases -- estirar una pose quieta no simula nada."""
    if item["label"] != "fall":
        return False
    if item["origen"] == "caucafall":
        return True  # caucafall no separa "fallen", todo "fall" ahí ya es transición
    return item.get("subclase") == 1  # 1 = fall (transición) en la taxonomía omnifall, 2 = fallen


def estirar_clip(kp, factor):
    """kp: (n_frames, 33, 4) = x, y, vx, vy ya normalizado (centrado en cadera, escalado).
    Interpolo x,y a más frames (misma trayectoria, más lenta) y recalculo la
    velocidad después de estirar -- estirar vx,vy directamente sería incorrecto,
    ya son una diferencia entre frames consecutivos que cambia de sentido al
    cambiar el espaciado temporal.
    """
    n = kp.shape[0]
    n_nuevo = int(round(n * factor))
    xy = kp[:, :, :2]

    t_original = np.linspace(0, 1, n)
    t_nuevo = np.linspace(0, 1, n_nuevo)

    xy_estirado = np.empty((n_nuevo, xy.shape[1], 2), dtype=xy.dtype)
    for p in range(xy.shape[1]):
        xy_estirado[:, p, 0] = np.interp(t_nuevo, t_original, xy[:, p, 0])
        xy_estirado[:, p, 1] = np.interp(t_nuevo, t_original, xy[:, p, 1])

    vel = np.zeros_like(xy_estirado)
    vel[1:] = xy_estirado[1:] - xy_estirado[:-1]

    return np.concatenate([xy_estirado, vel], axis=2)


if __name__ == "__main__":
    items = list(np.load(ENTRADA, allow_pickle=True))
    candidatos = [d for d in items if es_caida_transicion(d)]
    print(f"{len(items)} secuencias totales, {len(candidatos)} caídas de transición para estirar")

    sinteticos = []
    for item in candidatos:
        for factor in FACTORES_ESTIRAMIENTO:
            nuevo = dict(item)
            nuevo["keypoints"] = estirar_clip(item["keypoints"], factor)
            nuevo["subclase"] = "desmayo_sintetico"
            nuevo["origen"] = item["origen"] + "_desmayo_sintetico"
            nuevo["archivo_origen"] = f"{item['archivo_origen']}_x{factor}"
            # mismo "grupo" que el original a propósito: si el sujeto queda afuera
            # en un fold de la validación cruzada, su versión estirada tiene que
            # quedar afuera también, si no la CV por sujeto deja de servir.
            sinteticos.append(nuevo)

    print(f"{len(sinteticos)} clips sintéticos generados ({len(FACTORES_ESTIRAMIENTO)} por caída)")

    todos_con_desmayo = items + sinteticos
    np.save(SALIDA, todos_con_desmayo, allow_pickle=True)
    print(f"Guardado en {SALIDA} ({len(todos_con_desmayo)} secuencias totales)")
