import { useState, useMemo } from "react";

const DATOS_INICIALES = [
  {
    id: 1, tipo: "sector", nombre: "Sector1",
    camaras: [
      { id: 101, nombre: "Camara 5" },
      { id: 102, nombre: "Camara 6" },
      { id: 103, nombre: "Camara 7" },
      { id: 104, nombre: "Camara 8" },
    ],
  },
  { id: 2, tipo: "sector", nombre: "Sector2",
    camaras: [
      { id: 201, nombre: "Camara 1" },
      { id: 202, nombre: "Camara 2" },
      { id: 203, nombre: "Camara 3" },
      { id: 204, nombre: "Camara 4" },
    ],
  },
  { id: 3, tipo: "camara", nombre: "Camara 9" },  // suelta
];

export function useCamaras() {
  const [items, setItems]           = useState(DATOS_INICIALES);
  const [query, setQuery]           = useState("");
  const [editandoId, setEditandoId] = useState(null); // "s-1" | "c-101" | null
  const [previewId, setPreviewId]   = useState(null); // id de cámara en preview

  // ── Búsqueda ────────────────────────────────────────────────
  // Filtra sectores (por nombre de sector o nombre de cámara interna)
  // y cámaras sueltas. Un sector aparece si su nombre o alguna cámara coincide.
  const itemsFiltrados = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;

    return items.reduce((acc, item) => {
      if (item.tipo === "sector") {
        const camarasFiltradas = item.camaras.filter((c) =>
          c.nombre.toLowerCase().includes(q)
        );
        const sectorCoincide = item.nombre.toLowerCase().includes(q);
        if (sectorCoincide || camarasFiltradas.length > 0) {
          acc.push({
            ...item,
            camaras: sectorCoincide ? item.camaras : camarasFiltradas,
          });
        }
      } else {
        if (item.nombre.toLowerCase().includes(q)) acc.push(item);
      }
      return acc;
    }, []);
  }, [items, query]);

  // ── Edición de nombres ───────────────────────────────────────
  const toggleEdicion = (id) =>
    setEditandoId((prev) => (prev === id ? null : id));

  const guardarNombre = (id) => {
    setEditandoId(null);
    console.log("Guardar:", id, items);
  };

  const actualizarNombreSector = (id, valor) =>
    setItems((prev) =>
      prev.map((item) =>
        item.tipo === "sector" && item.id === id
          ? { ...item, nombre: valor }
          : item
      )
    );

  const actualizarNombreCamara = (sectorId, camaraId, valor) =>
    setItems((prev) =>
      prev.map((item) => {
        // Cámara suelta
        if (item.tipo === "camara" && item.id === camaraId) {
          return { ...item, nombre: valor };
        }
        // Cámara dentro de sector
        if (item.tipo === "sector" && item.id === sectorId) {
          return {
            ...item,
            camaras: item.camaras.map((c) =>
              c.id === camaraId ? { ...c, nombre: valor } : c
            ),
          };
        }
        return item;
      })
    );

  // ── Crear ────────────────────────────────────────────────────
  const crearSector = () => {
    const nuevoId = Date.now();
    setItems((prev) => [
      ...prev,
      { id: nuevoId, tipo: "sector", nombre: "Nuevo sector", camaras: [] },
    ]);
    setEditandoId(`s-${nuevoId}`);
    console.log("Crear sector");
  };

  const crearCamara = (sectorId = null) => {
    const nuevoId = Date.now();
    if (sectorId === null) {
      // Cámara suelta
      setItems((prev) => [
        ...prev,
        { id: nuevoId, tipo: "camara", nombre: "Nueva cámara" },
      ]);
      setEditandoId(`c-${nuevoId}`);
    } else {
      setItems((prev) =>
        prev.map((item) =>
          item.id === sectorId
            ? { ...item, camaras: [...item.camaras, { id: nuevoId, nombre: "Nueva cámara" }] }
            : item
        )
      );
      setEditandoId(`c-${nuevoId}`);
    }
    console.log("Crear cámara en sector:", sectorId);
  };

  // ── Preview ──────────────────────────────────────────────────
  const abrirPreview = (id) => setPreviewId(id);
  const cerrarPreview = () => setPreviewId(null);

  // ── Pinear ───────────────────────────────────────────────────
  const pinearCamara = (id) => {
    console.log("Pinear cámara:", id);
    // TODO: conectar con el grid de cámaras
  };

  return {
    items: itemsFiltrados,
    query,
    setQuery,
    editandoId,
    toggleEdicion,
    guardarNombre,
    actualizarNombreSector,
    actualizarNombreCamara,
    crearSector,
    crearCamara,
    previewId,
    abrirPreview,
    cerrarPreview,
    pinearCamara,
  };
}
