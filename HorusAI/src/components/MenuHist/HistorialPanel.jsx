import { useState } from "react";
import SearchBar from "./SearchBar/SearchBar";
import FilterButtons from "./FilterButtons/FilterButtons";
import EventoItem from "./EventoItem/EventoItem";
import VisorEvidencia from "./VisorEvidencia/VisorEvidencia";
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
  cargando = false,
  errorCarga = null,
  recargar,
  borrarHistorial,
  borrando = false,
  revisarAlerta,
}) {
  // Un historial vacio puede significar tres cosas MUY distintas: que todavia
  // no llego la respuesta, que el backend no contesta, o que de verdad no paso
  // nada. Mostrar "Sin resultados." en los tres casos es el mismo error que
  // una regla que se calla cuando no puede correr: el panel dice "todo
  // tranquilo" cuando en realidad no sabe.
  const hayFiltro = filtroActivo !== null || query.trim() !== "";
  const [viendo, setViendo] = useState(null);
  // Dos pasos en vez de un confirm(): el primer clic pregunta, el segundo
  // borra. Un historial no se vacia de un solo clic distraido.
  const [preguntando, setPreguntando] = useState(false);
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
              capturaUrl={ev.capturaUrl}
              clipUrl={ev.clipUrl}
              revision={ev.revision}
              onVer={() => setViendo(ev)}
            />
          ))}
        </ul>
      ) : errorCarga ? (
        <p className="historial-panel__empty historial-panel__empty--error">
          No se pudo traer el historial del backend ({errorCarga}).
          {recargar && (
            <>
              {" "}
              <button
                type="button"
                className="historial-panel__reintentar"
                onClick={() => recargar()}
              >
                Reintentar
              </button>
            </>
          )}
        </p>
      ) : cargando ? (
        <p className="historial-panel__empty">Cargando historial...</p>
      ) : hayFiltro ? (
        <p className="historial-panel__empty">Sin resultados para esta busqueda.</p>
      ) : (
        <p className="historial-panel__empty">Todavia no se registro ningun evento.</p>
      )}

      {borrarHistorial && eventosFiltrados.length > 0 && (
        <div className="historial-panel__limpiar">
          {!preguntando ? (
            <button type="button" className="historial-panel__reintentar"
                    onClick={() => setPreguntando(true)}>
              Limpiar historial
            </button>
          ) : (
            <>
              <span>¿Borrar todo el historial? Queda una copia.</span>
              <button type="button" className="historial-panel__reintentar"
                      disabled={borrando}
                      onClick={async () => { await borrarHistorial(); setPreguntando(false); }}>
                {borrando ? "Borrando..." : "Si, borrar"}
              </button>
              <button type="button" className="historial-panel__reintentar"
                      onClick={() => setPreguntando(false)}>
                No
              </button>
            </>
          )}
        </div>
      )}

      {viendo && (
        <VisorEvidencia
          evento={eventosFiltrados.find((e) => e.id === viendo.id) ?? viendo}
          onClose={() => setViendo(null)}
          onRevisar={revisarAlerta ? (v) => revisarAlerta(viendo.id, v) : null}
        />
      )}

    </div>
  );
}
