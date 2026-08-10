import { useState, useEffect, useRef } from "react";
import "./PreviewModal.css";

// Recibe una lista de cámaras del sector y navega entre ellas
export default function PreviewModal({ camaras = [], onClose }) {
  const [indice, setIndice] = useState(0);
  const imgRef = useRef(null);

  // Sin useEffect de cleanup — el stop lo maneja el handler de cierre
  const handleClose = () => {
    fetch('http://localhost:8000/video/stop_feed', { method: 'POST' })
      .catch(err => console.error('Error al detener backend:', err));
    onClose();
  };

  const anterior = () =>
    setIndice((prev) => (prev === 0 ? camaras.length - 1 : prev - 1));
  const siguiente = () =>
    setIndice((prev) => (prev === camaras.length - 1 ? 0 : prev + 1));

  const camara = camaras[indice];

  return (
    <div className="preview-modal__overlay" onClick={handleClose}>
      <div className="preview-modal" onClick={(e) => e.stopPropagation()}>

        {camaras.length > 1 && (
          <button className="preview-modal__arrow preview-modal__arrow--left" onClick={anterior}>❮</button>
        )}

        <div className="preview-modal__screen">
          <span className="preview-modal__label">{camara.nombre}</span>
          <img
            ref={imgRef}
            src={`http://localhost:8000/video/video_feed/${camara.id}?t=${Date.now()}`}
            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            alt={camara.nombre}
          />
        </div>

        {camaras.length > 1 && (
          <button className="preview-modal__arrow preview-modal__arrow--right" onClick={siguiente}>❯</button>
        )}

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

        <button className="preview-modal__close" onClick={handleClose}>✕</button>
      </div>
    </div>
  );
}