import { useState } from "react";
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
  onPreviewSector,  // abre el carrusel con todas las cámaras del sector
  onPinear,
}) {
  const idEdicionSector = `s-${sector.id}`;
  const editandoSector  = editandoId === idEdicionSector;

  // Cuando el sector está en edición, las cámaras internas
  // también muestran su input para editar nombre
  const camaraEnEdicion = (camaraId) =>
    editandoSector || editandoId === `c-${camaraId}`;

  return (
    <div className="sector-item">

      {/* ── Cabecera ── */}
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
            onClick={() =>
              editandoSector
                ? onGuardar(idEdicionSector)
                : onToggleEdicion(idEdicionSector)
            }
            active={editandoSector}
          />
          <IconButton
            icon={<IconOjo />}
            label="Ver cámaras del sector"
            onClick={() => onPreviewSector(sector)}
          />
          <IconButton
            icon={<IconPin />}
            label="Pinear sector"
            onClick={() => onPinear(sector.id)}
          />
        </div>
      </div>

      {/* ── Cámaras (sin botones propios) ── */}
      <div className="sector-item__camaras">
        {sector.camaras.map((camara) => (
          <CamaraItem
            key={camara.id}
            camara={camara}
            sectorId={sector.id}
            enSector={true}
            editando={camaraEnEdicion(camara.id)}
            onToggleEdicion={onToggleEdicion}
            onGuardar={onGuardar}
            onActualizarNombre={onActualizarNombreCamara}
            onPreview={() => {}}
            onPinear={() => {}}
          />
        ))}
      </div>

      <AddButton
        onClick={() => onCrearCamara(sector.id)}
        label="Agregar cámara al sector"
      />

    </div>
  );
}
