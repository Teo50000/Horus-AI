import { useState } from "react";
import SearchBar from "../MenuHist/SearchBar/SearchBar";
import SectorItem from "./SectorItem/SectorItem";
import CamaraItem from "./CamaraItem/CamaraItem";
import AddButton from "../MenuAjustes/AddButton/AddButton";
import PreviewModal from "./PreviewModal/PreviewModal";
import CloseButton from "../CloseButton/CloseButton";
import { useCamaras } from "./useCamaras";
import "./CamarasPanel.css";

export default function CamarasPanel({ onClose }) {
  const {
    items,
    query, setQuery,
    editandoId,
    toggleEdicion, guardarNombre,
    actualizarNombreSector, actualizarNombreCamara,
    crearSector, crearCamara,
    pinearCamara,
  } = useCamaras();

  // Preview: guarda la lista de cámaras a mostrar en el carrusel
  const [camarasPreview, setCamarasPreview] = useState([]);

  const abrirPreviewSector = (sector) => setCamarasPreview(sector.camaras);
  const abrirPreviewCamara = (camara)  => setCamarasPreview([camara]);
  const cerrarPreview = () => setCamarasPreview([]);

  return (
    <>
      <div className="camaras-panel" role="region" aria-label="Cámaras">

        <CloseButton onClick={onClose} />

        <SearchBar
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onClear={() => setQuery("")}
        />

        {/* Botón de acción principal — funcionalidad pendiente */}
        <button className="camaras-panel__action-btn" title="Acción principal">
          ⊞
        </button>

        <div className="camaras-panel__lista">
          {items.map((item) =>
            item.tipo === "sector" ? (
              <SectorItem
                key={item.id}
                sector={item}
                editandoId={editandoId}
                onToggleEdicion={toggleEdicion}
                onGuardar={guardarNombre}
                onActualizarNombreSector={actualizarNombreSector}
                onActualizarNombreCamara={actualizarNombreCamara}
                onCrearCamara={crearCamara}
                onPreviewSector={abrirPreviewSector}
                onPinear={pinearCamara}
              />
            ) : (
              // Cámara suelta — sí tiene sus propios botones
              <CamaraItem
                key={item.id}
                camara={item}
                sectorId={null}
                enSector={false}
                editando={editandoId === `c-${item.id}`}
                onToggleEdicion={toggleEdicion}
                onGuardar={guardarNombre}
                onActualizarNombre={actualizarNombreCamara}
                onPreview={() => abrirPreviewCamara(item)}
                onPinear={pinearCamara}
              />
            )
          )}
        </div>

        <AddButton onClick={crearSector} label="Crear sector" />

      </div>

      {camarasPreview.length > 0 && (
        <PreviewModal camaras={camarasPreview} onClose={cerrarPreview} />
      )}
    </>
  );
}
