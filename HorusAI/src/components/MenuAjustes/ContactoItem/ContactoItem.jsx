import EditButton from "../EditButton/EditButton";
import "./ContactoItem.css";

export default function contactoItem({ contacto, editando, onToggleEdicion, onActualizar, onGuardar }) {
  return (
    <div className="contacto-item">
      <div className="contacto-item__header">
        {editando ? (
          <input
            className="contacto-item__input contacto-item__input--nombre"
            value={contacto.nombre}
            onChange={(e) => onActualizar(contacto.id, "nombre", e.target.value)}
            placeholder="Nombre"
          />
        ) : (
          <span className="contacto-item__label">{contacto.nombre}</span>
        )}
        <EditButton
          modo={editando ? "guardar" : "editar"}
          onClick={() => editando ? onGuardar(contacto.id) : onToggleEdicion(contacto.id)}
        />
      </div>

      <input
        className="contacto-item__input"
        value={contacto.telefono}
        onChange={(e) => onActualizar(contacto.id, "telefono", e.target.value)}
        placeholder="ingresa tu email"
        disabled={!editando}
      />
    </div>
  );
}
