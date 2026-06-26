import EditButton from "../EditButton/EditButton";
import "./NumeroItem.css";

export default function NumeroItem({ numero, editando, onToggleEdicion, onActualizar, onGuardar }) {
  return (
    <div className="numero-item">
      <div className="numero-item__header">
        {editando ? (
          <input
            className="numero-item__input numero-item__input--nombre"
            value={numero.nombre}
            onChange={(e) => onActualizar(numero.id, "nombre", e.target.value)}
            placeholder="Nombre"
          />
        ) : (
          <span className="numero-item__label">{numero.nombre}</span>
        )}
        <EditButton
          modo={editando ? "guardar" : "editar"}
          onClick={() => editando ? onGuardar(numero.id) : onToggleEdicion(numero.id)}
        />
      </div>

      <input
        className="numero-item__input"
        value={numero.telefono}
        onChange={(e) => onActualizar(numero.id, "telefono", e.target.value)}
        placeholder="+549..."
        disabled={!editando}
      />
    </div>
  );
}
