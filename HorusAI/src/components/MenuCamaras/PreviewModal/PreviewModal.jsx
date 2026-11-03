import { useState } from "react";
import VideoStream from "../../VideoStream/VideoStream";
import "./PreviewModal.css";

export default function PreviewModal({
  camaras = [],
  onClose,
  urlBase = "http://localhost:8000/video/video_feed",
}) {
  const [indice, setIndice]       = useState(0);
  const [streamKey, setStreamKey] = useState(Date.now());

  if (!camaras.length) return null;

  const detenerStream = (camaraId) => {
    fetch(`http://localhost:8000/video/stop_feed/${camaraId}`, { method: "POST" })
      .catch((err) => console.error("Error al detener stream:", err));
  };

  const handleClose = () => {
    detenerStream(camaras[indice].id);
    onClose();
  };

  const anterior = () => {
    detenerStream(camaras[indice].id);
    setIndice((prev) => (prev === 0 ? camaras.length - 1 : prev - 1));
    setStreamKey(Date.now());
  };

  const siguiente = () => {
    detenerStream(camaras[indice].id);
    setIndice((prev) => (prev === camaras.length - 1 ? 0 : prev + 1));
    setStreamKey(Date.now());
  };

  const camara = camaras[indice];

  return (
    <div className="preview-modal__overlay" onClick={handleClose}>
      <div className="preview-modal" onClick={(e) => e.stopPropagation()}>

        {camaras.length > 1 && (
          <button className="preview-modal__arrow preview-modal__arrow--left" onClick={anterior}>❮</button>
        )}

        <div className="preview-modal__screen">
          <VideoStream
            key={streamKey}
            src={`${urlBase}/${camara.id}?t=${streamKey}`}
            alt={camara.nombre}
          />
          <span className="preview-modal__label">{camara.nombre}</span>
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
