import lapizIcon from "../../../assets/Lapiz.svg";
import "./EditButton.css";

export default function EditButton({ modo = "editar", onClick }) {
  return (
    <button
      className="edit-button"
      onClick={onClick}
      title={modo === "editar" ? "Editar" : "Guardar"}
      aria-label={modo === "editar" ? "Editar número" : "Guardar número"}
    >
      {modo === "editar"
        ? <img src={lapizIcon} alt="" />
        : <span className="edit-button__check">✓</span>
      }
    </button>
  );
}
