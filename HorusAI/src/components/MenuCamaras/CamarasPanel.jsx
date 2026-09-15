import { useState } from "react";
import SearchBar from "../MenuHist/SearchBar/SearchBar";
import SectorItem from "./SectorItem/SectorItem";
import CamaraItem from "./CamaraItem/CamaraItem";
import AddButton from "../MenuAjustes/AddButton/AddButton";
import RemoveButton from "../RemoveButton/RemoveButton";
import PreviewModal from "./PreviewModal/PreviewModal";
import CreacionModal from "./CreacionModal/CreacionModal";
import CloseButton from "../CloseButton/CloseButton";
import { useCamaras } from "./useCamaras";
import "./CamarasPanel.css";

export default function CamarasPanel({ onClose, onPinearCamara, onPinearSector }) {
  const {
    items, cargando, camarasSueltas,
    query, setQuery,
    editandoId, toggleEdicion, guardarNombre,
    actualizarNombreSector, actualizarNombreCamara,
    confirmarCreacion,
    modoBorrado, seleccionadosIds,
    toggleModoBorrado, toggleSeleccion,
    confirmarBorrado, cancelarBorrado,
  } = useCamaras();

  const [camarasPreview, setCamarasPreview]     = useState([]);
  const [modalConfig, setModalConfig]           = useState(null);
  const [previewHardware, setPreviewHardware]   = useState(null);

  const abrirPreviewSector = (sector) => setCamarasPreview(sector.camaras);
  const abrirPreviewCamara = (camara)  => setCamarasPreview([camara]);
  const cerrarPreview      = ()        => setCamarasPreview([]);

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
          {cargando ? (
            <p className="camaras-panel__cargando">Cargando cámaras...</p>
          ) : items.length === 0 ? (
            <p className="camaras-panel__vacio">No hay cámaras configuradas.</p>
          ) : (
            items.map((item) =>
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
                  onPinear={() => onPinearSector(item)}
                  // borrado
                  modoBorrado={modoBorrado}
                  seleccionadosIds={seleccionadosIds}
                  onToggleSeleccion={toggleSeleccion}
                />
              ) : (
                <div key={item.id} className="camaras-panel__camara-row">
                  {modoBorrado && (
                    <input
                      type="checkbox"
                      className="camaras-panel__checkbox"
                      checked={seleccionadosIds.has(item.id)}
                      onChange={() => toggleSeleccion(item.id)}
                    />
                  )}
                  <CamaraItem
                    camara={item}
                    sectorId={null}
                    enSector={false}
                    editando={!modoBorrado && editandoId === `c-${item.id}`}
                    onToggleEdicion={toggleEdicion}
                    onGuardar={guardarNombre}
                    onActualizarNombre={actualizarNombreCamara}
                    onPreview={() => abrirPreviewCamara(item)}
                    onPinear={() => onPinearCamara(item)}
                  />
                </div>
              )
            )
          )}
        </div>

        {/* Botones + y − / Confirmar y Cancelar */}
        <div className="camaras-panel__acciones">
          {modoBorrado ? (
            <>
              <button className="camaras-panel__cancelar" onClick={cancelarBorrado}>
                Cancelar
              </button>
              <button
                className="camaras-panel__confirmar"
                onClick={confirmarBorrado}
                disabled={seleccionadosIds.size === 0}
              >
                Aceptar
              </button>
            </>
          ) : (
            <>
              <AddButton onClick={abrirModalGeneral} label="Crear cámara o sector" />
              <RemoveButton onClick={toggleModoBorrado} label="Eliminar cámaras" />
            </>
          )}
        </div>
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
          urlBase="http://localhost:8000/video/preview"  // ← endpoint por usb_index
          onClose={() => setPreviewHardware(null)}
        />
      )}
    </>
  );
}
