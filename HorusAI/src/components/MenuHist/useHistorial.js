import { useState, useMemo } from "react";

/**
 * useHistorial
 * Encapsula toda la lógica del panel de historial:
 *  - apertura/cierre del panel
 *  - búsqueda por cámara o fecha
 *  - filtro por tipo de evento
 *  - generación de la lista filtrada
 *
 * Uso:
 *   const historial = useHistorial(eventos);
 *   <HistorialPanel {...historial} />
 */

// Tipos de evento disponibles como filtros
export const TIPOS_EVENTO = ["Desmayo", "Incendio", "Agresión"];

export function useHistorial(eventos = []) {
  // ── Estado del panel ──────────────────────────────────────────
  const [isOpen, setIsOpen] = useState(false);

  const togglePanel = () => setIsOpen((prev) => !prev);
  const closePanel = () => setIsOpen(false);

  // ── Estado de búsqueda ────────────────────────────────────────
  // El usuario puede escribir el nombre de una cámara ("Camara 3")
  // o una fecha ("12/06/25"). La búsqueda es case-insensitive.
  const [query, setQuery] = useState("");

  const handleQueryChange = (e) => setQuery(e.target.value);
  const clearQuery = () => setQuery("");

  // ── Estado de filtros ─────────────────────────────────────────
  // null  → sin filtro activo (muestra todos)
  // string → muestra sólo ese tipo de evento
  const [filtroActivo, setFiltroActivo] = useState(null);

  const toggleFiltro = (tipo) => {
    // Si clicás el mismo filtro que ya está activo, lo desactivás
    setFiltroActivo((prev) => (prev === tipo ? null : tipo));
  };

  // ── Lista filtrada ────────────────────────────────────────────
  // useMemo evita recalcular en cada render si no cambiaron las dependencias
  const eventosFiltrados = useMemo(() => {
    let resultado = eventos;

    // 1. Aplicar filtro de tipo
    if (filtroActivo !== null) {
      resultado = resultado.filter(
        (ev) => ev.tipo.toLowerCase() === filtroActivo.toLowerCase()
      );
    }

    // 2. Aplicar búsqueda (sobre los ya filtrados por tipo)
    const q = query.trim().toLowerCase();
    if (q !== "") {
      resultado = resultado.filter(
        (ev) =>
          ev.camara.toLowerCase().includes(q) ||
          ev.fecha.toLowerCase().includes(q)
      );
    }

    return resultado;
  }, [eventos, filtroActivo, query]);

  // ── Interfaz pública del hook ─────────────────────────────────
  return {
    // Panel
    isOpen,
    togglePanel,
    closePanel,

    // Búsqueda
    query,
    handleQueryChange,
    clearQuery,

    // Filtros
    filtroActivo,
    toggleFiltro,

    // Lista resultante
    eventosFiltrados,
  };
}
