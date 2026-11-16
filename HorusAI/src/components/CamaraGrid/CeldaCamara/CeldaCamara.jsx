import { useEffect, useRef, useState } from "react";
import VideoStream from "../../VideoStream/VideoStream";
import { API_URL, SERVICIO_URL } from "../../../config";
import "./CeldaCamara.css";

// Escalones de ancho que entiende el servicio (servicio.py `_ANCHOS`).
const ANCHOS = [320, 480, 640, 960, 1280, 1920];

/** El ancho de video que necesita una celda de la grilla 3x3, en píxeles
 *  físicos. Se calcula UNA vez al montar con la grilla en su tamaño máximo
 *  (menú cerrado): achicar la celda al abrir el menú no reconecta el video,
 *  solo lo escala. */
function anchoParaCelda() {
  if (typeof window === "undefined") return 640;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const celda = (window.innerWidth * 0.92) / 3;
  const necesita = celda * dpr;
  return ANCHOS.find((a) => a >= necesita) ?? 1280;
}

/**
 * De dónde sale el video de una cámara.
 *
 * 16/11: la celda le pedía SIEMPRE el video al backend, que para cada stream
 * preguntaba al servicio su /estado completo (con 3 s de timeout) y recién
 * después redirigía. Ahora va directo al servicio, que es el que tiene la
 * cámara abierta, y pide el tamaño justo para la celda. Si el servicio no
 * está corriendo, cae al video crudo del backend como antes.
 */
function urlVideo(camaraId, servicio, ancho) {
  if (!servicio?.consultado) return null;            // todavía no sé: skeleton
  // Mientras el servicio no esté AUSENTE (un reinicio, un tick lento) se le
  // sigue pidiendo a él: VideoStream reintenta solo. Ver useServicioVideo.
  if (servicio.activo || !servicio.ausente) {
    const cam = servicio.camaras?.[camaraId]?.camara ?? `cam-${camaraId}`;
    return `${SERVICIO_URL}/camaras/${cam}/stream?ancho=${ancho}&g=${servicio.generacion ?? 0}`;
  }
  return `${API_URL}/video/video_feed/${camaraId}`;
}

// Si el video sale del backend (servicio apagado), el backend tiene la
// cámara abierta: hay que avisarle que la suelte al sacarla de la grilla.
function soltarDelBackend(camaraId, servicio) {
  if (servicio?.activo) return;
  fetch(`${API_URL}/video/stop_feed/${camaraId}`, { method: "POST" })
    .catch(() => {});
}

function avisoDe(camaraId, servicio) {
  const c = servicio?.activo ? servicio.camaras?.[camaraId] : null;
  if (!c) return null;
  if (c.estado === "caida") return "Sin señal · reintentando";
  return null;
}

/** Cuando el video pasa del backend al servicio, el backend tiene que
 *  soltar la webcam: si no, el servicio recibe cuadros negros. */
function useSoltarAlCambiar(camaraId, url) {
  const anterior = useRef(url);
  useEffect(() => {
    const antes = anterior.current;
    anterior.current = url;
    if (camaraId == null || !antes || antes === url) return;
    if (antes.startsWith(`${API_URL}/video/`) && !(url ?? "").startsWith(`${API_URL}/video/`)) {
      fetch(`${API_URL}/video/stop_feed/${camaraId}`, { method: "POST" }).catch(() => {});
    }
  }, [camaraId, url]);
}

export default function CeldaCamara({ slot, slotIdx, onNavegar, onVaciar, onAgregar, servicio }) {
  const [ancho] = useState(anchoParaCelda);
  const idActual = slot
    ? (slot.tipo === "sector" ? slot.camaras[slot.indice]?.id : slot.id)
    : null;
  const urlActual = slot ? urlVideo(idActual, servicio, ancho) : null;
  useSoltarAlCambiar(idActual, urlActual);

  if (!slot) {
    return (
      <button
        type="button"
        className="celda-camara celda-camara--vacia"
        onClick={onAgregar}
        aria-label="Agregar una cámara a este espacio"
      >
        <span className="celda-camara__vacia-mas" aria-hidden="true">+</span>
        <span className="celda-camara__vacia-texto">Agregar cámara</span>
      </button>
    );
  }

  const esSector     = slot.tipo === "sector";
  const camaraActual = esSector ? slot.camaras[slot.indice] : slot;
  const hayVarias    = esSector && slot.camaras.length > 1;
  const conIA        = Boolean(servicio?.activo);

  return (
    <div className="celda-camara">
      <VideoStream
        key={camaraActual.id}
        src={urlActual}
        alt={camaraActual.nombre}
        aviso={avisoDe(camaraActual.id, servicio)}
      />

      <div className="celda-camara__pie">
        <span className="celda-camara__nombre" title={camaraActual.nombre}>
          {camaraActual.nombre}
        </span>
        {esSector && (
          <span className="celda-camara__sector-tag">
            {slot.nombre}{hayVarias ? ` · ${slot.indice + 1}/${slot.camaras.length}` : ""}
          </span>
        )}
      </div>

      {servicio?.consultado && (
        <span
          className={`celda-camara__ia ${conIA ? "" : "celda-camara__ia--no"}`}
          title={conIA
            ? "Los modelos están analizando esta cámara"
            : "El servicio de modelos no está corriendo: video sin análisis"}
        >
          {conIA ? "IA" : "SIN IA"}
        </span>
      )}

      <button
        type="button"
        className="celda-camara__unpin"
        onClick={() => { soltarDelBackend(camaraActual.id, servicio); onVaciar(slotIdx); }}
        title="Quitar de la grilla"
        aria-label={`Quitar ${camaraActual.nombre} de la grilla`}
      >✕</button>

      {hayVarias && (
        <>
          <button
            type="button"
            className="celda-camara__arrow celda-camara__arrow--left"
            onClick={() => { soltarDelBackend(camaraActual.id, servicio); onNavegar(slotIdx, "anterior"); }}
            aria-label="Cámara anterior del sector"
          >❮</button>
          <button
            type="button"
            className="celda-camara__arrow celda-camara__arrow--right"
            onClick={() => { soltarDelBackend(camaraActual.id, servicio); onNavegar(slotIdx, "siguiente"); }}
            aria-label="Cámara siguiente del sector"
          >❯</button>
        </>
      )}
    </div>
  );
}
