import "./EditButton.css";

// modo: "editar" | "guardar"
// El padre decide qué modo mostrar según si la fila está en edición
export default function EditButton({ modo = "editar", onClick }) {
  return (
    <button
      className="edit-button"
      onClick={onClick}
      title={modo === "editar" ? "Editar" : "Guardar"}
      aria-label={modo === "editar" ? "Editar número" : "Guardar número"}
    >
      {modo === "editar" ? "✏" : "✓"}
    </button>
  );
}
