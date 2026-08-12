import "./CeldaCamara.css";

const API = "http://localhost:8000";

const detenerStream = (camaraId) => {
  fetch(`${API}/video/stop_feed/${camaraId}`, { method: 'POST' })
    .catch(err => console.error('Error al detener stream:', err));
};

export default function CeldaCamara({ slot, slotIdx, onNavegar, onVaciar }) {
  if (!slot) {
    return <div className="celda-camara celda-camara--vacia" />;
  }
  if (slot.tipo === "camara") {
    return (
      <div className="celda-camara">
        <img
          src={`${API}/video/video_feed/${slot.id}?t=${Date.now()}`}
          className="celda-camara__stream"
          alt={slot.nombre}
        />
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
      <img
        key={`${slot.id}-${slot.indice}`}
        src={`${API}/video/video_feed/${camaraActual.id}?t=${Date.now()}`}
        alt={camaraActual.nombre}
        className="celda-camara__stream"
      />
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
