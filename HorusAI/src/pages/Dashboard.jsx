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
import "./Dashboard.css";

export default function Dashboard() {
  const [activeSection, setActiveSection] = useState(null); // null, no "null" string
  const [panelAbierto, setPanelAbierto]    = useState(null);

  // El hook va ACÁ ADENTRO, no afuera del componente
  const { eventos, conectado } = useWebSocketEventos("ws://localhost:8000/camaras/ws");

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
