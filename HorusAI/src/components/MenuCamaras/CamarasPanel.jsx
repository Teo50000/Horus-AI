import { useState } from "react";
import SearchBar from "../MenuHist/SearchBar/SearchBar";
import SectorItem from "./SectorItem/SectorItem";
import CamaraItem from "./CamaraItem/CamaraItem";
import AddButton from "../MenuAjustes/AddButton/AddButton";
import PreviewModal from "./PreviewModal/PreviewModal";
import CreacionModal from "./CreacionModal/CreacionModal";
import CloseButton from "../CloseButton/CloseButton";
import { useCamaras } from "./useCamaras";
import "./CamarasPanel.css";

export default function CamarasPanel({ onClose, onPinearCamara, onPinearSector }) {
  const {
    items,
    camarasSueltas,
    query, setQuery,
    editandoId,
    toggleEdicion, guardarNombre,
    actualizarNombreSector, actualizarNombreCamara,
    confirmarCreacion,
  } = useCamaras();

  // ── Preview carrusel ─────────────────────────────────────────
  const [camarasPreview, setCamarasPreview] = useState([]);
  const abrirPreviewSector = (sector) => setCamarasPreview(sector.camaras);
  const abrirPreviewCamara = (camara)  => setCamarasPreview([camara]);
  const cerrarPreview = () => setCamarasPreview([]);

  // ── Modal de creación ────────────────────────────────────────
  const [modalConfig, setModalConfig]       = useState(null);
  const [previewHardware, setPreviewHardware] = useState(null);

  const abrirModalGeneral  = ()       => setModalConfig({ modo: "camara" });
  const abrirModalAgregarA = (sector) => setModalConfig({ modo: "agregarASector", sector });
  const cerrarModal        = ()       => setModalConfig(null);

  const handleConfirmar = (resultado) => {
    confirmarCreacion(resultado);
    cerrarModal();
  };

  return (
    <>
      <div className="camaras-panel" role="region" aria-label="Cámaras">
        <CloseButton onClick={onClose} />

        <SearchBar
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onClear={() => setQuery("")}
        />

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
                onCrearCamara={() => abrirModalAgregarA(item)}
                onPreviewSector={abrirPreviewSector}
                onPinear={() => onPinearSector(item)}   // ← sector completo
              />
            ) : (
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
                onPinear={() => onPinearCamara(item)}   // ← cámara suelta
              />
            )
          )}
        </div>

        <AddButton onClick={abrirModalGeneral} label="Crear cámara o sector" />
      </div>

      {camarasPreview.length > 0 && (
        <PreviewModal camaras={camarasPreview} onClose={cerrarPreview} />
      )}

      {modalConfig && (
        <CreacionModal
          modoInicial={modalConfig.modo}
          sectorDestino={modalConfig.sector ?? null}
          camarasSueltas={camarasSueltas}
          onConfirmar={handleConfirmar}
          onCancelar={cerrarModal}
          onPreviewHardware={(cam) => setPreviewHardware(cam)}
        />
      )}

      {previewHardware && (
        <PreviewModal
          camaras={[{ id: previewHardware.id, nombre: previewHardware.nombre }]}
          onClose={() => setPreviewHardware(null)}
        />
      )}
    </>
  );
}
