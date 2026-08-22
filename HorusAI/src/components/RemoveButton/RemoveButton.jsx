import "./RemoveButton.css";

export default function RemoveButton({ onClick, activo = false, label = "Eliminar" }) {
  return (
    <button
      className={`remove-button ${activo ? "remove-button--activo" : ""}`}
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      −
    </button>
  );
}
