import "./PreviewModal.css";

// Placeholder de preview — reemplazar con feed real de cámara
export default function PreviewModal({ nombre, onClose }) {
  if (!nombre) return null;
  return (
    <div className="preview-modal__overlay" onClick={onClose}>
      <div className="preview-modal" onClick={(e) => e.stopPropagation()}>
        <div className="preview-modal__screen">
          <span className="preview-modal__label">Preview: {nombre}</span>
        </div>
        <button className="preview-modal__close" onClick={onClose}>✕</button>
      </div>
    </div>
  );
}
