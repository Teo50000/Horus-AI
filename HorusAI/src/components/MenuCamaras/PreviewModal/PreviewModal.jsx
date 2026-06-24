import { useState } from "react";
import "./PreviewModal.css";

// Recibe una lista de cámaras del sector y navega entre ellas
export default function PreviewModal({ camaras = [], onClose }) {
  const [indice, setIndice] = useState(0);

  if (!camaras.length) return null;

  const anterior = () =>
    setIndice((prev) => (prev === 0 ? camaras.length - 1 : prev - 1));
  const siguiente = () =>
    setIndice((prev) => (prev === camaras.length - 1 ? 0 : prev + 1));

  const camara = camaras[indice];

  return (
    <div className="preview-modal__overlay" onClick={onClose}>
      <div className="preview-modal" onClick={(e) => e.stopPropagation()}>

        {/* Flecha izquierda */}
        {camaras.length > 1 && (
          <button className="preview-modal__arrow preview-modal__arrow--left" onClick={anterior}>
            ❮
          </button>
        )}

        {/* Pantalla de la cámara */}
        <div className="preview-modal__screen">
          <span className="preview-modal__label">{camara.nombre}</span>
          {/* TODO: reemplazar con feed real de la cámara */}
        </div>

        {/* Flecha derecha */}
        {camaras.length > 1 && (
          <button className="preview-modal__arrow preview-modal__arrow--right" onClick={siguiente}>
            ❯
          </button>
        )}

        {/* Indicadores de posición */}
        {camaras.length > 1 && (
          <div className="preview-modal__dots">
            {camaras.map((_, i) => (
              <button
                key={i}
                className={`preview-modal__dot ${i === indice ? "preview-modal__dot--active" : ""}`}
                onClick={() => setIndice(i)}
              />
            ))}
          </div>
        )}

        <button className="preview-modal__close" onClick={onClose}>✕</button>
      </div>
    </div>
  );
}
