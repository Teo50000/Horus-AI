import IconButton from "../IconButton/IconButton";
import lapizIcon from "../../../assets/Lapiz.svg";
import "./CamaraItem.css";

// Ojo y pin como texto hasta que tengas los SVGs
const IconOjo  = () => <span style={{ fontSize: 12, color: "#fff" }}>👁</span>;
const IconPin  = () => <span style={{ fontSize: 12, color: "#fff" }}>📌</span>;

export default function CamaraItem({
  camara,
  sectorId = null,   // null si es cámara suelta
  editando,
  onToggleEdicion,
  onGuardar,
  onActualizarNombre,
  onPreview,
  onPinear,
}) {
  const idEdicion = `c-${camara.id}`;

  return (
    <div className="camara-item">
      {editando ? (
        <input
          className="camara-item__input"
          value={camara.nombre}
          onChange={(e) => onActualizarNombre(sectorId, camara.id, e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onGuardar(idEdicion)}
          autoFocus
        />
      ) : (
        <span className="camara-item__nombre">{camara.nombre}</span>
      )}

      <div className="camara-item__acciones">
        <IconButton
          icon={lapizIcon}
          label={editando ? "Guardar" : "Editar"}
          onClick={() => editando ? onGuardar(idEdicion) : onToggleEdicion(idEdicion)}
          active={editando}
        />
        <IconButton
          icon={<IconOjo />}
          label="Preview"
          onClick={() => onPreview(camara.id)}
        />
        <IconButton
          icon={<IconPin />}
          label="Pinear"
          onClick={() => onPinear(camara.id)}
        />
      </div>
    </div>
  );
}
