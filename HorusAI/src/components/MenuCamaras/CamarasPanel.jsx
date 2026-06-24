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
    previewId, abrirPreview, cerrarPreview,
    pinearCamara,
  } = useCamaras();

  // Encuentra el nombre de la cámara en preview
  const nombrePreview = (() => {
    for (const item of items) {
      if (item.tipo === "camara" && item.id === previewId) return item.nombre;
      if (item.tipo === "sector") {
        const c = item.camaras.find((c) => c.id === previewId);
        if (c) return c.nombre;
      }
    }
    return null;
  })();

  return (
    <>
      <div className="camaras-panel" role="region" aria-label="Cámaras">

        <CloseButton onClick={onClose} />

        {/* Barra de búsqueda — mismo componente que Historial */}
        <SearchBar
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onClear={() => setQuery("")}
        />

        {/* Botón de acción principal (funcionalidad pendiente) */}
        <button className="camaras-panel__action-btn" title="Acción principal">
          ⊞
        </button>

        {/* Lista de sectores y cámaras sueltas */}
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
                onPreview={abrirPreview}
                onPinear={pinearCamara}
              />
            ) : (
              <CamaraItem
                key={item.id}
                camara={item}
                sectorId={null}
                editando={editandoId === `c-${item.id}`}
                onToggleEdicion={toggleEdicion}
                onGuardar={guardarNombre}
                onActualizarNombre={actualizarNombreCamara}
                onPreview={abrirPreview}
                onPinear={pinearCamara}
              />
            )
          )}
        </div>

        {/* Crear sector nuevo */}
        <AddButton onClick={crearSector} label="Crear sector" />

      </div>

      {/* Preview fuera del panel para que no quede recortado */}
      {previewId && (
        <PreviewModal nombre={nombrePreview} onClose={cerrarPreview} />
      )}
    </>
  );
}
