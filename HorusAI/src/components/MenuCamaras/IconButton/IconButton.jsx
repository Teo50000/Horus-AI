import "./IconButton.css";

// Botón de ícono pequeño reutilizable para las acciones de sector/cámara
// icon: string (url de svg) o elemento JSX
export default function IconButton({ icon, onClick, label, active = false }) {
  return (
    <button
      className={`icon-button ${active ? "icon-button--active" : ""}`}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      {typeof icon === "string"
        ? <img src={icon} alt="" className="icon-button__img" />
        : icon
      }
    </button>
  );
}
