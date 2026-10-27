import numpy as np

# Preprocesado de keypoints de MediaPipe compartido por entrenamiento (03_normalizar),
# evaluacion offline y despliegue (12_webcam / detector_caidas). Una sola implementacion
# para que train y serve no puedan divergir.

IDX_CADERA_IZQ, IDX_CADERA_DER = 23, 24
IDX_HOMBRO_IZQ, IDX_HOMBRO_DER = 11, 12
IDX_CI, IDX_CD, IDX_HI, IDX_HD = IDX_CADERA_IZQ, IDX_CADERA_DER, IDX_HOMBRO_IZQ, IDX_HOMBRO_DER

UMBRAL_VISIBILIDAD = 0.5


# ---------------- offline (secuencia completa, puede mirar al futuro) ----------------

def interpolar_frames_faltantes(kp):
    """Rellena frames sin deteccion (todo en cero) interpolando desde vecinos validos.
    NO es causal: usa el frame valido siguiente. Solo para preparar datos de entrenamiento."""
    n_frames = kp.shape[0]
    valido = kp.reshape(n_frames, -1).sum(axis=1) != 0
    if valido.sum() == 0:
        return kp

    indices_validos = np.where(valido)[0]
    kp_interpolado = kp.copy()
    for i in range(n_frames):
        if not valido[i]:
            antes = indices_validos[indices_validos < i]
            despues = indices_validos[indices_validos > i]
            if len(antes) > 0 and len(despues) > 0:
                a, d = antes[-1], despues[0]
                peso = (i - a) / (d - a)
                kp_interpolado[i] = kp[a] * (1 - peso) + kp[d] * peso
            elif len(antes) > 0:
                kp_interpolado[i] = kp[antes[-1]]
            elif len(despues) > 0:
                kp_interpolado[i] = kp[despues[0]]
    return kp_interpolado


def normalizar_esqueleto(kp):
    """(n_frames,33,4) -> (n_frames,33,2). Centra en la cadera y escala por la
    distancia hombro-cadera, frame a frame."""
    cadera = (kp[:, IDX_CADERA_IZQ, :2] + kp[:, IDX_CADERA_DER, :2]) / 2
    hombro = (kp[:, IDX_HOMBRO_IZQ, :2] + kp[:, IDX_HOMBRO_DER, :2]) / 2
    escala = np.linalg.norm(hombro - cadera, axis=1, keepdims=True)
    escala = np.where(escala < 1e-6, 1e-6, escala)
    kp_norm = kp[:, :, :2] - cadera[:, None, :]
    return kp_norm / escala[:, None, :]


def agregar_velocidades(kp_norm):
    """(n_frames,33,2) -> (n_frames,33,4) con (x, y, dx, dy). El primer frame tiene dx=dy=0."""
    vel = np.zeros_like(kp_norm)
    vel[1:] = kp_norm[1:] - kp_norm[:-1]
    return np.concatenate([kp_norm, vel], axis=2)


# ---------------- online (un frame, causal por definicion) ----------------

def normalizar_frame(kp):
    """Normaliza UN frame: (33,4) -> (33,2). Misma transformacion que
    normalizar_esqueleto pero sin depender de otros frames."""
    cad = (kp[IDX_CI, :2] + kp[IDX_CD, :2]) / 2
    hom = (kp[IDX_HI, :2] + kp[IDX_HD, :2]) / 2
    esc = np.linalg.norm(hom - cad)
    esc = esc if esc > 1e-6 else 1e-6
    return (kp[:, :2] - cad) / esc


def visibilidad_confiable(kp):
    """Promedio de 'visibility' de cadera/hombro. Un recorte cortado en el borde de
    cuadro (persona saliendo, o acostada y semi-ocluida) suele devolver landmarks
    igual, pero con visibility baja en estos puntos clave."""
    vis = kp[[IDX_CI, IDX_CD, IDX_HI, IDX_HD], 3]
    return vis.mean() >= UMBRAL_VISIBILIDAD
