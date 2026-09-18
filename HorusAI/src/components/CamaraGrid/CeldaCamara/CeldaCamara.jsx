import { fuenteVideo } from "../fuenteVideo";
import "./CeldaCamara.css";

import { API_URL } from "../../../config";
const API = API_URL;

const detenerStream = (camaraId) => {
  fetch(`${API}/video/stop_feed/${camaraId}?t=${Date.now()}`, { method: 'POST' })
    .catch(err => console.error('Error al detener stream:', err));
};

function Stream({ camaraId, nombre, servicio, claveExtra }) {
  const f = fuenteVideo(camaraId, servicio);
  const caida = f.conIA && f.estado && f.estado !== "ok";

  return (
    <>
      <img
        key={`${camaraId}-${f.conIA}-${claveExtra ?? ""}`}
        src={f.url}
        className="celda-camara__stream"
        alt={nombre}
      />
      {f.conIA && (
        <span className="celda-camara__ia" title="Video con las detecciones dibujadas">
          IA
        </span>
      )}
      {/* Una camara caida y una camara donde no pasa nada se ven igual: negro.
          Este cartel es para que no se confundan. */}
      {caida && (
        <span className="celda-camara__caida">
          {f.estado}{f.error ? ` · ${f.error}` : ""}
        </span>
      )}
    </>
  );
}

export default function CeldaCamara({ slot, slotIdx, onNavegar, onVaciar, servicio }) {
  if (!slot) {
    return <div className="celda-camara celda-camara--vacia" />;
  }
  if (slot.tipo === "camara") {
    return (
      <div className="celda-camara">
        <Stream camaraId={slot.id} nombre={slot.nombre} servicio={servicio} />
        <span className="celda-camara__nombre">{slot.nombre}</span>
        <button
          className="celda-camara__unpin"
          onClick={() => { detenerStream(slot.id); onVaciar(slotIdx); }}
          title="Quitar"
        >
          ✕
        </button>
      </div>
    );
  }

  // Slot con sector (carrusel)
  const camaraActual = slot.camaras[slot.indice];
  const hayVarias    = slot.camaras.length > 1;

  return (
    <div className="celda-camara">
      <Stream camaraId={camaraActual.id} nombre={camaraActual.nombre}
              servicio={servicio} claveExtra={slot.indice} />
      <span className="celda-camara__nombre">{camaraActual.nombre}</span>
      <span className="celda-camara__sector-tag">{slot.nombre}</span>

      <button
        className="celda-camara__unpin"
        onClick={() => { detenerStream(slot.id); onVaciar(slotIdx); }}
        title="Quitar"
      >
        ✕
      </button>

      {hayVarias && (
        <>
          <button
            className="celda-camara__arrow celda-camara__arrow--left"
            onClick={() => {
              detenerStream(camaraActual.id);
              onNavegar(slotIdx, "anterior");
            }}
          >
            ❮
          </button>
          <button
            className="celda-camara__arrow celda-camara__arrow--right"
            onClick={() => {
              detenerStream(camaraActual.id);
              onNavegar(slotIdx, "siguiente");
            }}
          >
            ❯
          </button>
        </>
      )}
    </div>
  );
}
