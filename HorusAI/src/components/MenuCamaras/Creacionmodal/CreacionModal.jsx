import { useState } from "react";
import "./CreacionModal.css";

// ── Cámaras de hardware disponibles (simuladas) ───────────────
const HARDWARE_CAMARAS = [
  { id: "hw-1", nombre: "WebCamLogitechK20dhs" },
  { id: "hw-2", nombre: "ADMk43" },
  { id: "hw-3", nombre: "HITechcamweb" },
  { id: "hw-4", nombre: "Blablabla" },
];

// ── Selector de pestaña ───────────────────────────────────────
function TabSelector({ tab, onChange }) {
  return (
    <div className="creacion-modal__tabs">
      <button
        className={`creacion-modal__tab ${tab === "camara" ? "creacion-modal__tab--active" : ""}`}
        onClick={() => onChange("camara")}
      >
        Cámara
      </button>
      <button
        className={`creacion-modal__tab ${tab === "sector" ? "creacion-modal__tab--active" : ""}`}
        onClick={() => onChange("sector")}
      >
        Sector
      </button>
    </div>
  );
}

// ── Fila de cámara hardware seleccionable ─────────────────────
function HardwareCamaraRow({ camara, seleccionada, onToggle, onPreview }) {
  return (
    <div
      className={`creacion-modal__row ${seleccionada ? "creacion-modal__row--selected" : ""}`}
      onClick={onToggle}  // ← esto faltaba
    >
      <span className="creacion-modal__row-nombre">{camara.nombre}</span>
      <button
        className="creacion-modal__row-preview"
        onClick={(e) => {
          e.stopPropagation(); // ← evita que el click del preview también seleccione la fila
          onPreview(camara);
        }}
        title="Preview"
      >
        👁
      </button>
    </div>
  );
}

// ── Fila de cámara existente con checkbox ─────────────────────
function CamaraCheckRow({ camara, checked, onToggle }) {
  return (
    <label className={`creacion-modal__row creacion-modal__row--check ${checked ? "creacion-modal__row--selected" : ""}`}>
      <span className="creacion-modal__row-nombre">{camara.nombre}</span>
      <input
        type="checkbox"
        className="creacion-modal__checkbox"
        checked={checked}
        onChange={() => onToggle(camara.id)}
      />
    </label>
  );
}

// ── Modo: nueva cámara ────────────────────────────────────────
function ModoCamara({ onConfirmar, onCancelar, onPreview }) {
  const [nombre, setNombre]         = useState("Nueva Cámara");
  const [seleccionada, setSeleccion] = useState(null);

  const confirmar = () => {
    if (!seleccionada) return;
    onConfirmar({ tipo: "camara", nombre, hardwareId: seleccionada });
  };

  return (
    <>
      {/* Nombre editable */}
      <div className="creacion-modal__nombre-row">
        <input
          className="creacion-modal__nombre-input"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          placeholder="Nombre de la cámara"
        />
        <span className="creacion-modal__nombre-icon">✏</span>
      </div>

      {/* Lista de hardware */}
      <div className="creacion-modal__lista">
        {HARDWARE_CAMARAS.map((cam) => (
          <HardwareCamaraRow
            key={cam.id}
            camara={cam}
            seleccionada={seleccionada === cam.id}
            onToggle={() => setSeleccion((prev) => prev === cam.id ? null : cam.id)}
            onPreview={onPreview}
          />
        ))}
      </div>

      <div className="creacion-modal__footer">
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>
          Cancelar
        </button>
        <button
          className="creacion-modal__btn creacion-modal__btn--confirmar"
          onClick={confirmar}
          disabled={!seleccionada}
        >
          Aceptar
        </button>
      </div>
    </>
  );
}

// ── Modo: nuevo sector ────────────────────────────────────────
function ModoSector({ camarasSueltas, onConfirmar, onCancelar }) {
  const [nombre, setNombre]       = useState("Nuevo Sector");
  const [seleccionadas, setSelec] = useState(new Set());

  const toggleCamara = (id) =>
    setSelec((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const confirmar = () => {
    onConfirmar({
      tipo: "sector",
      nombre,
      camaraIds: [...seleccionadas],
    });
  };

  return (
    <>
      {/* Nombre editable */}
      <div className="creacion-modal__nombre-row">
        <input
          className="creacion-modal__nombre-input"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          placeholder="Nombre del sector"
        />
        <span className="creacion-modal__nombre-icon">✏</span>
      </div>

      {/* Lista de cámaras sueltas con checkbox */}
      <div className="creacion-modal__lista">
        {camarasSueltas.length === 0 && (
          <p className="creacion-modal__empty">No hay cámaras sueltas disponibles.</p>
        )}
        {camarasSueltas.map((cam) => (
          <CamaraCheckRow
            key={cam.id}
            camara={cam}
            checked={seleccionadas.has(cam.id)}
            onToggle={toggleCamara}
          />
        ))}
      </div>

      <div className="creacion-modal__footer">
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>
          Cancelar
        </button>
        <button
          className="creacion-modal__btn creacion-modal__btn--confirmar"
          onClick={confirmar}
        >
          Agregar
        </button>
      </div>
    </>
  );
}

// ── Modo: agregar cámaras a sector existente ──────────────────
function ModoAgregarASector({ sector, camarasSueltas, onConfirmar, onCancelar }) {
  const [seleccionadas, setSelec] = useState(new Set());

  const toggleCamara = (id) =>
    setSelec((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  const confirmar = () => {
    onConfirmar({
      tipo: "agregarASector",
      sectorId: sector.id,
      camaraIds: [...seleccionadas],
    });
  };

  return (
    <>
      <p className="creacion-modal__subtitulo">
        Agregar cámaras a <strong>{sector.nombre}</strong>
      </p>

      <div className="creacion-modal__lista">
        {camarasSueltas.length === 0 && (
          <p className="creacion-modal__empty">No hay cámaras sueltas disponibles.</p>
        )}
        {camarasSueltas.map((cam) => (
          <CamaraCheckRow
            key={cam.id}
            camara={cam}
            checked={seleccionadas.has(cam.id)}
            onToggle={toggleCamara}
          />
        ))}
      </div>

      <div className="creacion-modal__footer">
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>
          Cancelar
        </button>
        <button
          className="creacion-modal__btn creacion-modal__btn--confirmar"
          onClick={confirmar}
        >
          Agregar
        </button>
      </div>
    </>
  );
}

// ── Modal principal ───────────────────────────────────────────
// modoInicial: "camara" | "sector" | "agregarASector"
// sectorDestino: objeto sector (solo cuando modoInicial === "agregarASector")
export default function CreacionModal({
  modoInicial = "camara",
  sectorDestino = null,
  camarasSueltas = [],
  onConfirmar,
  onCancelar,
  onPreviewHardware,
}) {
  const [tab, setTab] = useState(modoInicial === "sector" ? "sector" : "camara");

  // En modo "agregarASector" no mostramos el selector de pestañas
  const esAgregarASector = modoInicial === "agregarASector";

  return (
    <div className="creacion-modal__overlay" onClick={onCancelar}>
      <div className="creacion-modal" onClick={(e) => e.stopPropagation()}>

        {/* Tabs — solo si no es "agregar a sector existente" */}
        {!esAgregarASector && (
          <TabSelector tab={tab} onChange={setTab} />
        )}

        {/* Contenido según modo */}
        {esAgregarASector ? (
          <ModoAgregarASector
            sector={sectorDestino}
            camarasSueltas={camarasSueltas}
            onConfirmar={onConfirmar}
            onCancelar={onCancelar}
          />
        ) : tab === "camara" ? (
          <ModoCamara
            onConfirmar={onConfirmar}
            onCancelar={onCancelar}
            onPreview={onPreviewHardware}
          />
        ) : (
          <ModoSector
            camarasSueltas={camarasSueltas}
            onConfirmar={onConfirmar}
            onCancelar={onCancelar}
          />
        )}

      </div>
    </div>
  );
}
