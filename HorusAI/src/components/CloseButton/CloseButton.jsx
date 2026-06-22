import "./CloseButton.css";

export default function CloseButton({ onClick }) {
  return (
    <button className="close-button" onClick={onClick} title="Cerrar panel">
      ‹
    </button>
  );
}
