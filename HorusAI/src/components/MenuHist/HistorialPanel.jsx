import SearchBar from "./SearchBar/SearchBar";
import FilterButtons from "./FilterButtons/FilterButtons";
import EventoItem from "./EventoItem/EventoItem";
import CloseButton from "../CloseButton/CloseButton";
import { TIPOS_EVENTO } from "./useHistorial";
import "./HistorialPanel.css";

export default function HistorialPanel({
  onClose,
  query,
  handleQueryChange,
  clearQuery,
  filtroActivo,
  toggleFiltro,
  eventosFiltrados,
}) {
  return (
    <div className="historial-panel" role="region" aria-label="Historial">

      <CloseButton onClick={onClose} />

      <SearchBar
        value={query}
        onChange={handleQueryChange}
        onClear={clearQuery}
      />

      <FilterButtons
        tipos={TIPOS_EVENTO}
        filtroActivo={filtroActivo}
        onToggle={toggleFiltro}
      />

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
