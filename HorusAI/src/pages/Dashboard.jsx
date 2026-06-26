import { useState } from "react";
import Sidebar from "../components/Sidebar/Sidebar";
import HistorialPanel from "../components/MenuHist/HistorialPanel";
import AjustesPanel from "../components/MenuAjustes/AjustesPanel";
import CamarasPanel from "../components/MenuCamaras/CamarasPanel";
import { useHistorial } from "../components/MenuHist/useHistorial";
import { useAjustes } from "../components/MenuAjustes/useAjustes";
import "./Dashboard.css";

const EVENTOS_EJEMPLO = [
  { id: 1, camara: "Camara 1", tipo: "Desmayo",  fecha: "12/06/25" },
  { id: 2, camara: "Camara 3", tipo: "Desmayo",  fecha: "11/06/25" },
  { id: 3, camara: "Camara 2", tipo: "Incendio", fecha: "10/06/25" },
  { id: 4, camara: "Camara 5", tipo: "Agresión", fecha: "09/06/25" },
  { id: 5, camara: "Camara 1", tipo: "Agresión", fecha: "08/06/25" },
];

export default function Dashboard() {
  const [activeSection, setActiveSection]   = useState("null");
  const [panelAbierto, setPanelAbierto]     = useState(null); // "camaras" | "historial" | "ajustes" | null

  const historial = useHistorial(EVENTOS_EJEMPLO);
  const ajustes   = useAjustes();

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
    setActiveSection("null");
  };

  return (
    <div className="dashboard">

      <Sidebar
        activeSection={activeSection}
        onSectionChange={handleSectionChange}
      />

      {panelAbierto === "camaras"   && <CamarasPanel  onClose={cerrarPanel} />}
      {panelAbierto === "historial" && <HistorialPanel {...historial} onClose={cerrarPanel} />}
      {panelAbierto === "ajustes"   && <AjustesPanel  {...ajustes}   onClose={cerrarPanel} />}

      <main className="dashboard-main">
        {/* Grid de cámaras */}
      </main>

    </div>
  );
}
