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
import { WS_ALERTAS } from "../config";
import "./Dashboard.css";

export default function Dashboard() {
  const [activeSection, setActiveSection] = useState(null); // null, no "null" string
  const [panelAbierto, setPanelAbierto]    = useState(null);

  // El hook va ACÁ ADENTRO, no afuera del componente
  const { eventos, conectado } = useWebSocketEventos(WS_ALERTAS);

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
      />

    </div>
  );
}
