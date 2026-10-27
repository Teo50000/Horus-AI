import { useState, useEffect } from "react";
import "./CreacionModal.css";
import iconPreview from "../../../assets/PreView.svg";
import { API_VIDEO } from "../../../config";

// ── Marcas de cámaras IP ─────────────────────────────────────
// TODO: completar con las marcas que aceptan
const MARCAS_IP = [
  { value: "", label: "Seleccionar marca..." },
  // { value: "hikvision", label: "Hikvision" },
  // { value: "dahua",     label: "Dahua" },
  // { value: "axis",      label: "Axis" },
];

// ── Endpoint de cámara IP ────────────────────────────────────
// TODO: reemplazar con el endpoint real cuando esté disponible
const API_CAMARA_IP = null; // ej: `${API_CAMARAS}/config/ip`

// ── Selector de pestaña ──────────────────────────────────────
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

// ── Tabs internas de ModoCamara ──────────────────────────────
function SubTabSelector({ tab, onChange }) {
  return (
    <div className="creacion-modal__subtabs">
      <button
        className={`creacion-modal__subtab ${tab === "web" ? "creacion-modal__subtab--active" : ""}`}
        onClick={() => onChange("web")}
      >
        Cámara web
      </button>
      <button
        className={`creacion-modal__subtab ${tab === "ip" ? "creacion-modal__subtab--active" : ""}`}
        onClick={() => onChange("ip")}
      >
        Cámara IP
      </button>
    </div>
  );
}

// ── Fila de cámara hardware seleccionable ────────────────────
function HardwareCamaraRow({ camara, seleccionada, onToggle, onPreview }) {
  return (
    <div
      className={`creacion-modal__row ${seleccionada ? "creacion-modal__row--selected" : ""}`}
      onClick={onToggle}
    >
      <span className="creacion-modal__row-nombre">{camara.nombre}</span>
      <button
        className="creacion-modal__row-preview"
        onClick={(e) => { e.stopPropagation(); onPreview(camara); }}
        title="Preview"
      >
        <img src={iconPreview} alt="Preview" className="creacion-modal__row-preview-img" />
      </button>
    </div>
  );
}

// ── Fila de cámara existente con checkbox ────────────────────
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

// ── Formulario cámara web ────────────────────────────────────
function FormWeb({ nombre, setNombre, onConfirmar, onCancelar, onPreview }) {
  const [seleccionada, setSeleccion]       = useState(null);
  const [camarasDisponibles, setCamaras]   = useState([]);
  const [buscando, setBuscando]            = useState(true);
  const [motivo, setMotivo]                = useState(null);

  const buscar = (forzar = false) => {
    setBuscando(true);
    fetch(`${API_VIDEO}/cameras/available${forzar ? "?refrescar=true" : ""}`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        setMotivo(res.headers.get("X-Horus-Motivo"));
        return res.headers.get("X-Horus-Buscando")
          ? res.json().then((d) => { setTimeout(() => buscar(false), 2500); return d; })
          : res.json();
      })
      .then((data) => setCamaras(Array.isArray(data) ? data : []))
      .catch((err) => {
        setCamaras([]);
        setMotivo(`error conectando al backend (${err.message})`);
      })
      .finally(() => setBuscando(false));
  };

  useEffect(() => { buscar(false); }, []);

  const confirmar = () => {
    if (seleccionada === null) return;
    onConfirmar({ tipo: "camara", nombre, hardwareId: seleccionada, rtspUrl: null });
  };

  return (
    <>
      <div className="creacion-modal__nombre-row">
        <input
          className="creacion-modal__nombre-input"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          placeholder="Nombre de la cámara"
        />
      </div>

      <div className="creacion-modal__lista">
        {camarasDisponibles.map((cam) => (
          <HardwareCamaraRow
            key={cam.usb_index}
            camara={{ id: cam.usb_index, nombre: cam.nombre + (cam.resolucion ? ` · ${cam.resolucion}` : "") }}
            seleccionada={seleccionada === cam.usb_index}
            onToggle={() => setSeleccion((prev) => prev === cam.usb_index ? null : cam.usb_index)}
            onPreview={onPreview}
          />
        ))}

        {camarasDisponibles.length === 0 && (
          <div className="creacion-modal__vacio">
            {buscando ? "Buscando cámaras..." : (
              <>
                <div>No se encontró ninguna cámara.</div>
                {motivo && <div className="creacion-modal__motivo">{motivo}</div>}
                <div className="creacion-modal__ayuda">
                  En Windows la webcam la abre un programa a la vez: si la
                  tenés abierta en Zoom, Discord o la app Cámara, cerralos.
                </div>
                <button type="button" className="creacion-modal__reintentar" onClick={() => buscar(true)}>
                  Buscar de nuevo
                </button>
              </>
            )}
          </div>
        )}
      </div>

      <div className="creacion-modal__footer">
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>Cancelar</button>
        <button
          className="creacion-modal__btn creacion-modal__btn--confirmar"
          onClick={confirmar}
          disabled={seleccionada === null}
        >
          Aceptar
        </button>
      </div>
    </>
  );
}

// ── Formulario cámara IP ─────────────────────────────────────
function FormIP({ nombre, setNombre, onConfirmar, onCancelar }) {
  const [usuario, setUsuario]   = useState("");
  const [password, setPassword] = useState("");
  const [ip, setIp]             = useState("");
  const [marca, setMarca]       = useState("");

  const valido = usuario.trim() && password.trim() && ip.trim() && marca;

  const confirmar = async () => {
    if (!valido) return;

    if (!API_CAMARA_IP) {
      // TODO: conectar cuando el endpoint esté disponible
      console.log("Cámara IP — datos a enviar:", { nombre, usuario, password, ip, marca });
      onConfirmar({ tipo: "camara", nombre, rtspUrl: `pendiente-${ip}`, hardwareId: null });
      return;
    }

    try {
      const res = await fetch(API_CAMARA_IP, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nombre, usuario, password, ip, marca }),
      });
      if (!res.ok) throw new Error(`Error ${res.status}`);
      const data = await res.json();
      onConfirmar({ tipo: "camara", nombre, rtspUrl: data.rtsp_url, hardwareId: null });
    } catch (err) {
      console.error("Error al agregar cámara IP:", err);
    }
  };

  return (
    <>
      <div className="creacion-modal__nombre-row">
        <input
          className="creacion-modal__nombre-input"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          placeholder="Nombre de la cámara"
        />
      </div>

      <div className="creacion-modal__ip-form">
        <input
          className="creacion-modal__ip-input"
          value={usuario}
          onChange={(e) => setUsuario(e.target.value)}
          placeholder="Usuario"
          autoComplete="off"
        />
        <input
          className="creacion-modal__ip-input"
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Contraseña"
          autoComplete="new-password"
        />
        <input
          className="creacion-modal__ip-input"
          value={ip}
          onChange={(e) => setIp(e.target.value)}
          placeholder="Dirección IP  (ej: 192.168.1.100)"
        />
        <select
          className="creacion-modal__ip-select"
          value={marca}
          onChange={(e) => setMarca(e.target.value)}
        >
          {MARCAS_IP.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </div>

      <div className="creacion-modal__footer">
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>Cancelar</button>
        <button
          className="creacion-modal__btn creacion-modal__btn--confirmar"
          onClick={confirmar}
          disabled={!valido}
        >
          Aceptar
        </button>
      </div>
    </>
  );
}

// ── Modo: nueva cámara (orquesta web / IP) ───────────────────
function ModoCamara({ onConfirmar, onCancelar, onPreview }) {
  const [subTab, setSubTab] = useState("web");
  const [nombre, setNombre] = useState("Nueva Cámara");

  return (
    <>
      <SubTabSelector tab={subTab} onChange={setSubTab} />

      {subTab === "web" ? (
        <FormWeb
          nombre={nombre}
          setNombre={setNombre}
          onConfirmar={onConfirmar}
          onCancelar={onCancelar}
          onPreview={onPreview}
        />
      ) : (
        <FormIP
          nombre={nombre}
          setNombre={setNombre}
          onConfirmar={onConfirmar}
          onCancelar={onCancelar}
        />
      )}
    </>
  );
}

// ── Modo: nuevo sector ───────────────────────────────────────
function ModoSector({ camarasSueltas, onConfirmar, onCancelar }) {
  const [nombre, setNombre]       = useState("Nuevo Sector");
  const [seleccionadas, setSelec] = useState(new Set());

  const toggleCamara = (id) =>
    setSelec((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

  return (
    <>
      <div className="creacion-modal__nombre-row">
        <input
          className="creacion-modal__nombre-input"
          value={nombre}
          onChange={(e) => setNombre(e.target.value)}
          placeholder="Nombre del sector"
        />
      </div>
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
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>Cancelar</button>
        <button className="creacion-modal__btn creacion-modal__btn--confirmar" onClick={() => onConfirmar({ tipo: "sector", nombre, camaraIds: [...seleccionadas] })}>
          Agregar
        </button>
      </div>
    </>
  );
}

// ── Modo: agregar cámaras a sector existente ─────────────────
function ModoAgregarASector({ sector, camarasSueltas, onConfirmar, onCancelar }) {
  const [seleccionadas, setSelec] = useState(new Set());

  const toggleCamara = (id) =>
    setSelec((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });

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
        <button className="creacion-modal__btn creacion-modal__btn--cancelar" onClick={onCancelar}>Cancelar</button>
        <button className="creacion-modal__btn creacion-modal__btn--confirmar" onClick={() => onConfirmar({ tipo: "agregarASector", sectorId: sector.id, camaraIds: [...seleccionadas] })}>
          Agregar
        </button>
      </div>
    </>
  );
}

// ── Modal principal ──────────────────────────────────────────
export default function CreacionModal({
  modoInicial = "camara",
  sectorDestino = null,
  camarasSueltas = [],
  onConfirmar,
  onCancelar,
  onPreviewHardware,
}) {
  const [tab, setTab] = useState(modoInicial === "sector" ? "sector" : "camara");
  const esAgregarASector = modoInicial === "agregarASector";

  return (
    <div className="creacion-modal__overlay" onClick={onCancelar}>
      <div className="creacion-modal" onClick={(e) => e.stopPropagation()}>
        {!esAgregarASector && <TabSelector tab={tab} onChange={setTab} />}

        {esAgregarASector ? (
          <ModoAgregarASector sector={sectorDestino} camarasSueltas={camarasSueltas} onConfirmar={onConfirmar} onCancelar={onCancelar} />
        ) : tab === "camara" ? (
          <ModoCamara onConfirmar={onConfirmar} onCancelar={onCancelar} onPreview={onPreviewHardware} />
        ) : (
          <ModoSector camarasSueltas={camarasSueltas} onConfirmar={onConfirmar} onCancelar={onCancelar} />
        )}
      </div>
    </div>
  );
}
