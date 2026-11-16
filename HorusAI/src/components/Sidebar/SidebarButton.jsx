import "./SidebarButton.css";

export default function SidebarButton({ icon, label, active, onClick }) {
  return (
    <button
      type="button"
      className={`sidebar-button ${active ? "active" : ""}`}
      onClick={onClick}
      title={label}
      aria-label={label}
      aria-pressed={active}
    >
      <img src={icon} alt="" />
    </button>
  );
}
