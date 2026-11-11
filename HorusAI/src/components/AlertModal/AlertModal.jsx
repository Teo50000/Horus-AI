import { useEffect } from "react";
import "./AlertModal.css";

function formatearHora(fecha) {
  const d = new Date(fecha);
  if (isNaN(d)) return fecha; // si no se puede parsear, mostramos el valor crudo
  return d.toLocaleTimeString("es-AR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

// alerta: { tipo, camara, fecha } | null
// Invisible mientras alerta sea null.
export default function AlertModal({ alerta, onClose }) {
  // Escape cierra el modal
  useEffect(() => {
    if (!alerta) return;
    const onKey = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [alerta, onClose]);

  if (!alerta) return null;

  return (
    <div className="alert-modal__overlay">
      <div
        className="alert-modal"
        role="alertdialog"
        aria-live="assertive"
        aria-label={`Alerta de ${alerta.tipo}`}
      >
        <button
          className="alert-modal__close"
          onClick={onClose}
          aria-label="Cerrar alerta"
        >
          ✕
        </button>

        <div className="alert-modal__icon" aria-hidden="true">⚠</div>

        <span className="alert-modal__label">Alerta detectada</span>
        <h2 className="alert-modal__tipo">{alerta.tipo}</h2>

        <div className="alert-modal__detalle">
          <div className="alert-modal__fila">
            <span className="alert-modal__fila-label">Cámara</span>
            <span className="alert-modal__fila-valor">{alerta.camara}</span>
          </div>
          <div className="alert-modal__fila">
            <span className="alert-modal__fila-label">Hora</span>
            <span className="alert-modal__fila-valor">{formatearHora(alerta.fecha)}</span>
          </div>
        </div>

        <button className="alert-modal__btn" onClick={onClose}>
          Cerrar
        </button>
      </div>
    </div>
  );
}
