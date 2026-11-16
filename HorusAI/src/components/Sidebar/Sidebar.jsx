import SidebarButton from "./SidebarButton";
import EstadoSistema from "./EstadoSistema";

import logo from "../../assets/Logo.svg";
import cameraIcon from "../../assets/Cámaras.svg";
import settingsIcon from "../../assets/Ajustes.svg";
import historyIcon from "../../assets/Historial.svg";

import "./Sidebar.css";

// Sidebar stateless: solo recibe qué sección está activa y el callback
export default function Sidebar({
  activeSection, onSectionChange, onAyuda, conectado, servicio, mail,
}) {
  return (
    <aside className="sidebar">

      <img src={logo} alt="HorusAI" className="sidebar-logo" />

      <nav className="sidebar-buttons" aria-label="Menú">
        <SidebarButton
          icon={cameraIcon}
          label="Cámaras"
          active={activeSection === "cameras"}
          onClick={() => onSectionChange("cameras")}
        />
        <SidebarButton
          icon={settingsIcon}
          label="Ajustes"
          active={activeSection === "settings"}
          onClick={() => onSectionChange("settings")}
        />
        <SidebarButton
          icon={historyIcon}
          label="Historial"
          active={activeSection === "history"}
          onClick={() => onSectionChange("history")}
        />
      </nav>

      <EstadoSistema conectado={conectado} servicio={servicio} mail={mail} />

      {/* 16/11: era la palabra "Ayuda" en naranja, sin forma de botón. El
          signo de pregunta en un círculo es la convención que todos
          reconocen, y ocupa lo mismo en cualquier ancho de sidebar. */}
      <button
        type="button"
        className="help-button"
        onClick={onAyuda}
        title="Ayuda"
        aria-label="Ayuda"
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
             strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
      </button>

    </aside>
  );
}
