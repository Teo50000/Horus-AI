import "./FilterButtons.css";

// Recibe la lista de filtros disponibles, cuál está activo, y el handler
// tipos: array de strings, ej: ["Desmayo", "Incendio", "Agresión"]
export default function FilterButtons({ tipos, filtroActivo, onToggle }) {
  return (
    <div className="filter-buttons" role="group" aria-label="Filtrar por tipo">
      {tipos.map((tipo) => (
        <button
          key={tipo}
          className={`filter-buttons__btn ${
            filtroActivo === tipo ? "filter-buttons__btn--active" : ""
          }`}
          onClick={() => onToggle(tipo)}
          aria-pressed={filtroActivo === tipo}
        >
          {tipo + "s"}
        </button>
      ))}
    </div>
  );
}
