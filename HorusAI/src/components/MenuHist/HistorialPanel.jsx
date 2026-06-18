import { TIPOS_EVENTO } from "./useHistorial";
import "./HistorialPanel.css";

// ── EventoItem ────────────────────────────────────────────────
// Renderiza una fila de la lista a partir de un objeto evento.
// Estructura del objeto:
// { id, camara: "Camara 3", tipo: "Desmayo", fecha: "12/06/25" }
function EventoItem({ camara, tipo, fecha }) {
  return (
    <li className="historial-panel__item">
      <span className="historial-panel__item-camara">{camara}:</span>
      <span className="historial-panel__item-tipo">{tipo}</span>
      <span className="historial-panel__item-fecha">{fecha}</span>
    </li>
  );
}

// ── HistorialPanel ────────────────────────────────────────────
// Recibe el estado y los handlers directamente desde useHistorial.
// No tiene lógica propia; sólo presenta.
function HistorialPanel({
  isOpen,
  query,
  handleQueryChange,
  clearQuery,
  filtroActivo,
  toggleFiltro,
  eventosFiltrados,
}) {
  if (!isOpen) return null;

  return (
    <div className="historial-panel" role="region" aria-label="Historial">
      {/* Barra de búsqueda */}
      <div className="historial-panel__search">
        <input
          type="text"
          className="historial-panel__input"
          placeholder="Buscar"
          value={query}
          onChange={handleQueryChange}
          aria-label="Buscar en historial"
        />
        {query && (
          <button
            className="historial-panel__clear"
            onClick={clearQuery}
            aria-label="Limpiar búsqueda"
          >
            ✕
          </button>
        )}
        <span className="historial-panel__search-icon" aria-hidden="true">
          🔍
        </span>
      </div>

      {/* Filtros de tipo de evento */}
      <div className="historial-panel__filters" role="group" aria-label="Filtrar por tipo">
        {TIPOS_EVENTO.map((tipo) => (
          <button
            key={tipo}
            className={`historial-panel__filter-btn ${
              filtroActivo === tipo ? "historial-panel__filter-btn--active" : ""
            }`}
            onClick={() => toggleFiltro(tipo)}
            aria-pressed={filtroActivo === tipo}
          >
            {tipo + "s"}
          </button>
        ))}
      </div>

      {/* Lista de eventos */}
      {eventosFiltrados.length > 0 ? (
        <ul className="historial-panel__list">
          {eventosFiltrados.map((ev) => (
            <EventoItem
              key={ev.id}
              camara={ev.camara}
              tipo={ev.tipo}
              fecha={ev.fecha}
            />
          ))}
        </ul>
      ) : (
        <p className="historial-panel__empty">Sin resultados.</p>
      )}
    </div>
  );
}

export default HistorialPanel;
