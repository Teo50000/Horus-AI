import { useState } from "react";
import "./VideoStream.css";

export default function VideoStream({ src, alt = "" }) {
  const [error, setError] = useState(false);

  return (
    <div className="video-stream">
      {error ? (
        <div className="video-stream__error">
          <span className="video-stream__error-icon">⚠</span>
          <span className="video-stream__error-text">Sin señal</span>
        </div>
      ) : (
        <>
          {/* Spinner siempre visible detrás del video */}
          <div className="video-stream__loader">
            <div className="video-stream__spinner" />
            <span className="video-stream__loader-text">Conectando...</span>
          </div>

          {/* El video va encima y tapa el spinner cuando llega */}
          <img
            src={src}
            alt={alt}
            className="video-stream__img"
            onError={() => setError(true)}
          />
        </>
      )}
    </div>
  );
}
