import "./CloseButton.css";
import CloseIcon from "../../assets/CloseSimbol.svg"

export default function CloseButton({ onClick }) {
  return (
    <button className="close-button" onClick={onClick} title="Cerrar panel">
      <img src={CloseIcon} alt="Cerrar" className="close__icon " />
    </button>
  );
}
