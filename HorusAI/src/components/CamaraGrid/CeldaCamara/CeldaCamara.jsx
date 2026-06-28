import "./CeldaCamara.css";

export default function CeldaCamara({ slot, slotIdx, onNavegar, onVaciar }) {
  // Slot vacío
  if (!slot) {
    return <div className="celda-camara celda-camara--vacia" />;
  }

  // Slot con cámara suelta
  if (slot.tipo === "camara") {
    return (
      <div className="celda-camara">
        <span className="celda-camara__nombre">{slot.nombre}</span>
        <button
          className="celda-camara__unpin"
          onClick={() => onVaciar(slotIdx)}
          title="Quitar"
        >
          ✕
        </button>
        {/* TODO: reemplazar con feed de video */}
      </div>
    );
  }

  // Slot con sector (carrusel)
  const camaraActual = slot.camaras[slot.indice];
  const hayVarias    = slot.camaras.length > 1;

  return (
    <div className="celda-camara">
      <span className="celda-camara__nombre">{camaraActual.nombre}</span>
      <span className="celda-camara__sector-tag">{slot.nombre}</span>

      <button
        className="celda-camara__unpin"
        onClick={() => onVaciar(slotIdx)}
        title="Quitar"
      >
        ✕
      </button>

      {hayVarias && (
        <>
          <button
            className="celda-camara__arrow celda-camara__arrow--left"
            onClick={() => onNavegar(slotIdx, "anterior")}
          >
            ❮
          </button>
          <button
            className="celda-camara__arrow celda-camara__arrow--right"
            onClick={() => onNavegar(slotIdx, "siguiente")}
          >
            ❯
          </button>
        </>
      )}
      {/* TODO: reemplazar con feed de video */}
    </div>
  );
}
