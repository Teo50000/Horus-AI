import { enPlural } from "../../../hooks/useWebSocketEventos";
import "./FilterButtons.css";

// Recibe la lista de filtros disponibles, cuál está activo, y el handler
// tipos: array de strings EN SINGULAR, ej: ["Agresión", "Caída", "Incendio"].
// El boton se ve en plural pero el valor que viaja al filtro es el singular,
// que es el que tienen los eventos.
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
          {enPlural(tipo)}
        </button>
      ))}
    </div>
  );
}
