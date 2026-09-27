import { useEffect, useState } from "react";
import { API_URL } from "../../../config";
import "./VisorEvidencia.css";

// La ruta que guarda el backend es relativa ("/evidencia/clips/..."). Se
// arma la absoluta aca, en un solo lugar.
export function urlEvidencia(ruta) {
  if (!ruta) return null;
  if (/^https?:\/\//.test(ruta)) return ruta;
  return `${API_URL}${ruta}`;
}

/**
 * VisorEvidencia
 *
 * 26/09: lo que se veia cuando salto la alerta. El clip son los segundos
 * ANTERIORES (lo que explica por que salto) y la captura, el cuadro del
 * momento con las cajas dibujadas. El clip puede ser video (.mp4/.webm) o un
 * GIF si la PC no tiene un codec de video que el navegador sepa reproducir.
 */
export default function VisorEvidencia({ evento, onClose, onRevisar }) {
  const [enviando, setEnviando] = useState(false);
  const [error, setError] = useState(null);
  const revisar = async (veredicto) => {
    if (!onRevisar) return;
    setEnviando(true);
    setError(null);
    try {
      // Tocar el mismo boton otra vez saca la marca.
      await onRevisar(evento.revision === veredicto ? "" : veredicto);
    } catch (e) {
      setError(e.message);
    } finally {
      setEnviando(false);
    }
  };
  useEffect(() => {
    const alTeclado = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", alTeclado);
    return () => window.removeEventListener("keydown", alTeclado);
  }, [onClose]);

  const captura = urlEvidencia(evento.capturaUrl);
  const clip = urlEvidencia(evento.clipUrl);
  const clipEsGif = clip && /\.gif($|\?)/i.test(clip);

  return (
    <div className="visor-evidencia__overlay" onClick={onClose}>
      <div className="visor-evidencia" onClick={(e) => e.stopPropagation()}
           role="dialog" aria-label={`Evidencia: ${evento.tipo}`}>
        <header className="visor-evidencia__cabecera">
          <span className="visor-evidencia__tipo">{evento.tipo}</span>
          <span className="visor-evidencia__donde">
            {evento.camara} · {evento.fecha}
          </span>
          <button className="visor-evidencia__cerrar" onClick={onClose}
                  title="Cerrar">✕</button>
        </header>

        <div className="visor-evidencia__medios">
          {clip && (
            <figure className="visor-evidencia__figura">
              {clipEsGif ? (
                <img src={clip} alt="Los segundos anteriores a la alerta" />
              ) : (
                <video src={clip} controls autoPlay loop muted playsInline />
              )}
              <figcaption>Los segundos anteriores</figcaption>
            </figure>
          )}
          {captura && (
            <figure className="visor-evidencia__figura">
              <img src={captura} alt="El momento de la alerta" />
              <figcaption>El momento de la alerta</figcaption>
            </figure>
          )}
        </div>

        <footer className="visor-evidencia__pie">
          {captura && <a href={captura} target="_blank" rel="noreferrer" download>Bajar la imagen</a>}
          {clip && <a href={clip} target="_blank" rel="noreferrer" download>Bajar el video</a>}

          {onRevisar && (
            <div className="visor-evidencia__revision">
              <span>¿Fue de verdad?</span>
              <button type="button" disabled={enviando}
                      className={`visor-evidencia__voto ${evento.revision === "correcta" ? "visor-evidencia__voto--si" : ""}`}
                      onClick={() => revisar("correcta")}>
                ✓ Fue real
              </button>
              <button type="button" disabled={enviando}
                      className={`visor-evidencia__voto ${evento.revision === "falsa" ? "visor-evidencia__voto--no" : ""}`}
                      onClick={() => revisar("falsa")}>
                ✗ Falsa alarma
              </button>
              {evento.revision === "falsa" && (
                <span className="visor-evidencia__nota">Guardada para enseñarle al modelo</span>
              )}
              {error && <span className="visor-evidencia__nota">No se pudo guardar ({error})</span>}
            </div>
          )}
        </footer>
      </div>
    </div>
  );
}
