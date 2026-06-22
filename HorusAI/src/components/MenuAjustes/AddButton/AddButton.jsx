import "./AddButton.css";

export default function AddButton({ onClick, label = "Agregar" }) {
  return (
    <button
      className="add-button"
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      +
    </button>
  );
}
