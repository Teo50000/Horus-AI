import "./AddButton.css";
import Add from "../../../assets/Add.svg"

export default function AddButton({ onClick, label = "Agregar" }) {
  return (
    <button
      className="add-button"
      onClick={onClick}
      title={label}
      aria-label={label}
    >
      <img className=".add-button img"src={Add}/>
    </button>
  );
}
