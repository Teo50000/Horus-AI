import { useState } from "react";
import Sidebar from "../components/Sidebar/Sidebar";
import HistorialPanel from "../components/MenuHist/HistorialPanel";
import AjustesPanel from "../components/MenuAjustes/AjustesPanel";
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
  const [activeSection, setActiveSection] = useState("cameras");

  const historial = useHistorial(EVENTOS_EJEMPLO);
  const ajustes   = useAjustes();

  // Abre solo el panel correspondiente, cierra el otro
  const [panelAbierto, setPanelAbierto] = useState(null); // "historial" | "ajustes" | null

  const handleSectionChange = (section) => {
    setActiveSection(section);

    if (section === "history") {
      setPanelAbierto((prev) => (prev === "historial" ? null : "historial"));
    } else if (section === "settings") {
      setPanelAbierto((prev) => (prev === "ajustes" ? null : "ajustes"));
    } else {
      setPanelAbierto(null);
    }
  };

  const cerrarPanel = () => {
    setPanelAbierto(null);
    setActiveSection("cameras");
  };

  return (
    <div className="dashboard">

      <Sidebar
        activeSection={activeSection}
        onSectionChange={handleSectionChange}
      />

      {panelAbierto === "historial" && (
        <HistorialPanel {...historial} onClose={cerrarPanel} />
      )}

      {panelAbierto === "ajustes" && (
        <AjustesPanel {...ajustes} isOpen onClose={cerrarPanel} />
      )}

      <main className="dashboard-main">
        {/* Cámaras */}
      </main>

    </div>
  );
}
