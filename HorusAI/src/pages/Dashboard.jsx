import { useCallback, useEffect, useState } from "react";
import Sidebar from "../components/Sidebar/Sidebar";
import HistorialPanel from "../components/MenuHist/HistorialPanel";
import AjustesPanel from "../components/MenuAjustes/AjustesPanel";
import CamarasPanel from "../components/MenuCamaras/CamarasPanel";
import CamaraGrid from "../components/CamaraGrid/CamaraGrid";
import AyudaModal from "../components/AyudaModal/AyudaModal";
import { useHistorial } from "../components/MenuHist/useHistorial";
import { useAjustes } from "../components/MenuAjustes/useAjustes";
import { useGrid } from "../components/CamaraGrid/useGrid";
import { useWebSocketEventos } from "../hooks/useWebSocketEventos";
import { useServicioVideo } from "../hooks/useServicioVideo";
import { useEstadoBackend } from "../hooks/useEstadoBackend";
import { WS_ALERTAS } from "../config";
import AlertModal from "../components/AlertModal/AlertModal";
import { useAlertaActiva } from "../hooks/useAlertaActiva";
import "./Dashboard.css";

const SECCION_A_PANEL = { cameras: "camaras", settings: "ajustes", history: "historial" };

export default function Dashboard() {
  const [panelAbierto, setPanelAbierto] = useState(null);   // "camaras" | "ajustes" | "historial" | null
  // El panel que se DIBUJA. Al cerrar queda montado hasta que termina la
  // animación, así se ve cerrarse en vez de desaparecer de golpe.
  const [panelMostrado, setPanelMostrado] = useState(null);
  const [ayudaAbierta, setAyudaAbierta] = useState(false);

  const { eventos, conectado, limpiar: limpiarEnVivo } = useWebSocketEventos(WS_ALERTAS);

  // El servicio de modelos: si esta corriendo, el video de las camaras sale
  // de el (con las cajas dibujadas) en vez de salir del backend. Ver el
  // comentario en useServicioVideo.
  const servicio = useServicioVideo();
  const { mail } = useEstadoBackend();

  const historial = useHistorial(eventos, limpiarEnVivo);
  const ajustes   = useAjustes();
  const { alerta, cerrarAlerta } = useAlertaActiva(eventos, ajustes.alertaEnPantalla);
  const grid      = useGrid();

  useEffect(() => {
    if (panelAbierto) setPanelMostrado(panelAbierto);
  }, [panelAbierto]);

  const handleSectionChange = (section) => {
    const destino = SECCION_A_PANEL[section] ?? null;
    setPanelAbierto((prev) => (prev === destino ? null : destino));
  };

  const cerrarPanel = useCallback(() => setPanelAbierto(null), []);
  const abrirCamaras = useCallback(() => setPanelAbierto("camaras"), []);
  const cerrarAyuda = useCallback(() => setAyudaAbierta(false), []);

  const activeSection = Object.keys(SECCION_A_PANEL)
    .find((k) => SECCION_A_PANEL[k] === panelAbierto) ?? null;

  return (
    <div className="dashboard">

      {/* 16/11: los tres carteles de estado ("En vivo", "Analizando N",
          "Alertas sin mail") estaban fijos arriba a la derecha, encima de la
          cámara de esa esquina. Ahora viven en el sidebar (EstadoSistema):
          siguen siempre a la vista y siguen latiendo si algo falla, pero no
          tapan video. Los motivos de por qué existen (18/09, 22/09) están
          en EstadoSistema.jsx. */}
      <Sidebar
        activeSection={activeSection}
        onSectionChange={handleSectionChange}
        onAyuda={() => setAyudaAbierta(true)}
        conectado={conectado}
        servicio={servicio}
        mail={mail}
      />

      {/* El menú desplegable. Anima su ANCHO (no solo se desliza encima), así
          la grilla de cámaras se reacomoda de forma continua mientras se
          abre o se cierra, en vez de saltar de tamaño en un cuadro. */}
      <div
        className={`dashboard__panel ${panelAbierto ? "dashboard__panel--abierto" : ""}`}
        onTransitionEnd={(e) => {
          if (e.target === e.currentTarget && e.propertyName === "width" && !panelAbierto) {
            setPanelMostrado(null);
          }
        }}
        aria-hidden={!panelAbierto}
      >
        <div className="dashboard__panel-interior">
          {panelMostrado === "camaras" && (
            <CamarasPanel
              onClose={cerrarPanel}
              onPinearCamara={grid.pinearCamara}
              onPinearSector={grid.pinearSector}
            />
          )}
          {panelMostrado === "historial" && (
            <HistorialPanel {...historial} onClose={cerrarPanel} />
          )}
          {panelMostrado === "ajustes" && (
            <AjustesPanel {...ajustes} onClose={cerrarPanel} />
          )}
        </div>
      </div>

      <CamaraGrid
        slots={grid.slots}
        onNavegar={grid.navegarSector}
        onVaciar={grid.vaciarSlot}
        onAgregar={abrirCamaras}
        servicio={servicio}
      />

      <AlertModal alerta={alerta} onClose={cerrarAlerta} />
      <AyudaModal abierto={ayudaAbierta} onClose={cerrarAyuda} />

    </div>
  );
}
