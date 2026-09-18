import { useState, useMemo, useEffect, useCallback } from "react";
import { API_ALERTAS } from "../../config";
import { TIPOS_EN_ESPANOL, eventoDeLaBase } from "../../hooks/useWebSocketEventos";

/**
 * useHistorial
 * Encapsula toda la logica del panel de historial:
 *  - carga de lo ya guardado en el backend
 *  - apertura/cierre del panel
 *  - busqueda por camara o fecha
 *  - filtro por tipo de evento
 *  - generacion de la lista filtrada
 *
 * Uso:
 *   const historial = useHistorial(eventos);
 *   <HistorialPanel {...historial} />
 */

// 18/09: esto era una lista escrita a mano con tres tipos —Desmayo, Incendio,
// Agresion— mientras la fusion emite diez. Ahora sale del mismo mapa que usa
// mapearTipo, asi que no se puede desincronizar.
export const TIPOS_EVENTO = TIPOS_EN_ESPANOL;

export function useHistorial(eventos = []) {
  // -- Estado del panel ------------------------------------------
  const [isOpen, setIsOpen] = useState(false);

  const togglePanel = () => setIsOpen((prev) => !prev);
  const closePanel = () => setIsOpen(false);

  // -- Lo que ya estaba guardado ---------------------------------
  // Hasta hoy el historial solo mostraba lo que habia llegado por websocket
  // DESDE que abriste la app: cerrabas el panel, lo volvias a abrir y seguia,
  // pero reiniciabas y arrancaba vacio aunque el backend tuviera todo en
  // horus.db. Eso hace que un incendio de anoche sea indistinguible de que no
  // haya pasado nada, que es justo lo que este sistema no puede permitirse.
  const [guardados, setGuardados] = useState([]);
  const [cargando, setCargando]   = useState(true);
  const [errorCarga, setErrorCarga] = useState(null);

  const recargar = useCallback(async (señal) => {
    setCargando(true);
    try {
      const r = await fetch(`${API_ALERTAS}?limite=200`, { signal: señal });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const filas = await r.json();
      setGuardados(filas.map(eventoDeLaBase));
      setErrorCarga(null);
    } catch (err) {
      if (err.name === "AbortError") return;
      // Que el panel siga andando sin backend es a proposito: se ve lo que
      // llega en vivo y el cartel avisa que lo viejo no se pudo traer. Lo que
      // no puede pasar es que un historial vacio parezca "no paso nada".
      console.warn("Historial: no se pudo cargar lo guardado:", err.message);
      setErrorCarga(err.message);
    } finally {
      setCargando(false);
    }
  }, []);

  useEffect(() => {
    const ac = new AbortController();
    recargar(ac.signal);
    return () => ac.abort();
  }, [recargar]);

  // -- Estado de busqueda ----------------------------------------
  // El usuario puede escribir el nombre de una camara ("Camara 3")
  // o una fecha ("12/06/25"). La busqueda es case-insensitive.
  const [query, setQuery] = useState("");

  const handleQueryChange = (e) => setQuery(e.target.value);
  const clearQuery = () => setQuery("");

  // -- Estado de filtros -----------------------------------------
  // null  -> sin filtro activo (muestra todos)
  // string -> muestra solo ese tipo de evento
  const [filtroActivo, setFiltroActivo] = useState(null);

  const toggleFiltro = (tipo) => {
    // Si clicas el mismo filtro que ya esta activo, lo desactivas
    setFiltroActivo((prev) => (prev === tipo ? null : tipo));
  };

  // -- En vivo + guardado, sin repetir ---------------------------
  // El que llego por websocket gana: es el mismo evento pero con el nombre de
  // camara resuelto y, si subio de severidad, con la severidad nueva.
  const todos = useMemo(() => {
    const porId = new Map();
    for (const ev of guardados) porId.set(ev.id, ev);
    for (const ev of eventos)   porId.set(ev.id, ev);
    // Se ordena por la fecha ISO, no por la corta: "24/08/26" no se ordena
    // bien como texto (el dia queda adelante del anio).
    return [...porId.values()].sort((a, b) =>
      String(b.fechaIso ?? "").localeCompare(String(a.fechaIso ?? ""))
    );
  }, [eventos, guardados]);

  // -- Lista filtrada --------------------------------------------
  // useMemo evita recalcular en cada render si no cambiaron las dependencias
  const eventosFiltrados = useMemo(() => {
    let resultado = todos;

    // 1. Aplicar filtro de tipo
    if (filtroActivo !== null) {
      resultado = resultado.filter(
        (ev) => String(ev.tipo).toLowerCase() === String(filtroActivo).toLowerCase()
      );
    }

    // 2. Aplicar busqueda (sobre los ya filtrados por tipo)
    const q = query.trim().toLowerCase();
    if (q !== "") {
      resultado = resultado.filter(
        (ev) =>
          String(ev.camara ?? "").toLowerCase().includes(q) ||
          String(ev.fecha ?? "").toLowerCase().includes(q)
      );
    }

    return resultado;
  }, [todos, filtroActivo, query]);

  // -- Interfaz publica del hook ---------------------------------
  return {
    // Panel
    isOpen,
    togglePanel,
    closePanel,

    // Busqueda
    query,
    handleQueryChange,
    clearQuery,

    // Filtros
    filtroActivo,
    toggleFiltro,

    // Lista resultante
    eventosFiltrados,

    // Carga del historial guardado
    cargando,
    errorCarga,
    recargar,
  };
}
