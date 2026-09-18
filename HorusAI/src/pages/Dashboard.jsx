import { useState } from "react";
import Sidebar from "../components/Sidebar/Sidebar";
import HistorialPanel from "../components/MenuHist/HistorialPanel";
import AjustesPanel from "../components/MenuAjustes/AjustesPanel";
import CamarasPanel from "../components/MenuCamaras/CamarasPanel";
import CamaraGrid from "../components/CamaraGrid/CamaraGrid";
import { useHistorial } from "../components/MenuHist/useHistorial";
import { useAjustes } from "../components/MenuAjustes/useAjustes";
import { useGrid } from "../components/CamaraGrid/useGrid";
import { useWebSocketEventos } from "../hooks/useWebSocketEventos";
import { useServicioVideo } from "../hooks/useServicioVideo";
import { WS_ALERTAS } from "../config";
import "./Dashboard.css";

export default function Dashboard() {
  const [activeSection, setActiveSection] = useState(null); // null, no "null" string
  const [panelAbierto, setPanelAbierto]    = useState(null);

  // El hook va ACÁ ADENTRO, no afuera del componente
  const { eventos, conectado } = useWebSocketEventos(WS_ALERTAS);

  // El servicio de modelos: si esta corriendo, el video de las camaras sale
  // de el (con las cajas dibujadas) en vez de salir del backend. Ver el
  // comentario en useServicioVideo.
  const servicio = useServicioVideo();

  const historial = useHistorial(eventos);
  const ajustes   = useAjustes();
  const grid      = useGrid();

  const handleSectionChange = (section) => {
    setActiveSection(section);
    setPanelAbierto((prev) => {
      if (section === "cameras")  return prev === "camaras"   ? null : "camaras";
      if (section === "settings") return prev === "ajustes"   ? null : "ajustes";
      if (section === "history")  return prev === "historial" ? null : "historial";
      return null;
    });
  };

  const cerrarPanel = () => {
    setPanelAbierto(null);
    setActiveSection(null);
  };

  return (
    <div className="dashboard">

      {/* Un panel de vigilancia que perdio el websocket se ve EXACTAMENTE
          igual que uno donde no pasa nada: pantalla tranquila, historial
          quieto. Es el mismo problema que una regla que se calla cuando no
          puede correr. El cartel esta para que "no hubo alertas" y "nadie las
          esta escuchando" nunca se parezcan. */}
      <div
        className={`dashboard__conexion ${
          conectado ? "dashboard__conexion--ok" : "dashboard__conexion--caida"
        }`}
        role="status"
        aria-live="polite"
        title={conectado ? WS_ALERTAS : `Sin conexion a ${WS_ALERTAS}`}
      >
        <span className="dashboard__conexion-punto" aria-hidden="true" />
        {conectado ? "En vivo" : "SIN CONEXION AL BACKEND"}
      </div>

      {/* Que las camaras se vean no quiere decir que algo las este mirando.
          Son dos procesos distintos: el backend sirve el video, el servicio
          corre los modelos. Sin este cartel, un panel con las camaras
          andando y los modelos apagados se ve igual que uno vigilando. */}
      <div
        className={`dashboard__modelos ${
          servicio.activo ? "dashboard__modelos--ok" : "dashboard__modelos--no"
        }`}
        role="status"
        title={servicio.activo
          ? `${Object.keys(servicio.camaras).length} camara(s) en analisis`
          : "El servicio de modelos no esta corriendo: nadie esta mirando el video"}
      >
        <span className="dashboard__conexion-punto" aria-hidden="true" />
        {servicio.activo
          ? `Analizando ${Object.keys(servicio.camaras).length}`
          : "MODELOS APAGADOS"}
      </div>

      <Sidebar
        activeSection={activeSection}
        onSectionChange={handleSectionChange}
      />

      {panelAbierto === "camaras" && (
        <CamarasPanel
          onClose={cerrarPanel}
          onPinearCamara={grid.pinearCamara}
          onPinearSector={grid.pinearSector}
        />
      )}
      {panelAbierto === "historial" && (
        <HistorialPanel {...historial} onClose={cerrarPanel} />
      )}
      {panelAbierto === "ajustes" && (
        <AjustesPanel {...ajustes} onClose={cerrarPanel} />
      )}

      <CamaraGrid
        slots={grid.slots}
        onNavegar={grid.navegarSector}
        onVaciar={grid.vaciarSlot}
        servicio={servicio}
      />

    </div>
  );
}
