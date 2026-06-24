import CamaraItem from "../CamaraItem/CamaraItem";
import AddButton from "../../MenuAjustes/AddButton/AddButton";
import IconButton from "../IconButton/IconButton";
import lapizIcon from "../../../assets/Lapiz.svg";
import "./SectorItem.css";

const IconOjo = () => <span style={{ fontSize: 12, color: "#fff" }}>👁</span>;
const IconPin = () => <span style={{ fontSize: 12, color: "#fff" }}>📌</span>;

export default function SectorItem({
  sector,
  editandoId,
  onToggleEdicion,
  onGuardar,
  onActualizarNombreSector,
  onActualizarNombreCamara,
  onCrearCamara,
  onPreview,
  onPinear,
}) {
  const idEdicionSector = `s-${sector.id}`;
  const editandoSector  = editandoId === idEdicionSector;

  return (
    <div className="sector-item">

      {/* ── Cabecera del sector ── */}
      <div className="sector-item__header">
        {editandoSector ? (
          <input
            className="sector-item__input"
            value={sector.nombre}
            onChange={(e) => onActualizarNombreSector(sector.id, e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onGuardar(idEdicionSector)}
            autoFocus
          />
        ) : (
          <span className="sector-item__nombre">{sector.nombre}</span>
        )}

        <div className="sector-item__acciones">
          <IconButton
            icon={lapizIcon}
            label={editandoSector ? "Guardar" : "Editar sector"}
            onClick={() => editandoSector ? onGuardar(idEdicionSector) : onToggleEdicion(idEdicionSector)}
            active={editandoSector}
          />
          <IconButton
            icon={<IconOjo />}
            label="Preview sector"
            onClick={() => onPreview(sector.id)}
          />
          <IconButton
            icon={<IconPin />}
            label="Pinear sector"
            onClick={() => onPinear(sector.id)}
          />
        </div>
      </div>

      {/* ── Cámaras del sector ── */}
      <div className="sector-item__camaras">
        {sector.camaras.map((camara) => (
          <CamaraItem
            key={camara.id}
            camara={camara}
            sectorId={sector.id}
            editando={editandoId === `c-${camara.id}`}
            onToggleEdicion={onToggleEdicion}
            onGuardar={onGuardar}
            onActualizarNombre={onActualizarNombreCamara}
            onPreview={onPreview}
            onPinear={onPinear}
          />
        ))}
      </div>

      {/* ── Agregar cámara al sector ── */}
      <AddButton
        onClick={() => onCrearCamara(sector.id)}
        label="Agregar cámara al sector"
      />

    </div>
  );
}
