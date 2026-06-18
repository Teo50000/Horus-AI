import SidebarButton from "./SidebarButton";

import logo from "../../assets/logo.svg";
import cameraIcon from "../../assets/Cámaras.svg";
import settingsIcon from "../../assets/Ajustes.svg";
import historyIcon from "../../assets/Historial.svg";

import "./Sidebar.css";

// Sidebar ya no tiene estado ni lógica de historial.
// Solo recibe qué sección está activa y callbacks del Dashboard.
export default function Sidebar({ activeSection, onSectionChange, onHistoryClick }) {

  const handleHistory = () => {
    onSectionChange("history");
    onHistoryClick();
  };

  return (
    <aside className="sidebar">

      <img
        src={logo}
        alt="HorusAI"
        className="sidebar-logo"
      />

      <div className="sidebar-buttons">

        <SidebarButton
          icon={cameraIcon}
          active={activeSection === "cameras"}
          onClick={() => onSectionChange("cameras")}
        />

        <SidebarButton
          icon={settingsIcon}
          active={activeSection === "settings"}
          onClick={() => onSectionChange("settings")}
        />

        <SidebarButton
          icon={historyIcon}
          active={activeSection === "history"}
          onClick={handleHistory}
        />

      </div>

      <button className="help-button">
        Ayuda
      </button>

    </aside>
  );
}