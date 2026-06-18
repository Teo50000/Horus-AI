export default function SidebarButton({ icon, active, onClick }) {
  return (
    <button
      className={`sidebar-button ${active ? "active" : ""}`}
      onClick={onClick}
    >
      <img src={icon} alt="" />
    </button>
  );
}