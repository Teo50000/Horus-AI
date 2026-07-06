import { useState, useMemo } from "react";

const DATOS_INICIALES = [
  {
    id: 1, tipo: "sector", nombre: "Sector1",
    camaras: [
      { id: 4, nombre: "Camara 1" },
      { id: 102, nombre: "Camara 6" },
      { id: 103, nombre: "Camara 7" },
      { id: 104, nombre: "Camara 8" },
    ],
  },
  { id: 3, tipo: "camara", nombre: "Camara 1" },
  { id: 4, tipo: "camara", nombre: "Camara 2" },
  { id: 5, tipo: "camara", nombre: "Camara 3" },
  { id: 6, tipo: "camara", nombre: "Camara 4" },
];

export function useCamaras() {
  const [items, setItems]           = useState(DATOS_INICIALES);
  const [query, setQuery]           = useState("");
  const [editandoId, setEditandoId] = useState(null);

  // ── Cámaras sueltas (sin sector) ────────────────────────────
  // Usadas por el modal de creación de sector y agregar a sector
  const camarasSueltas = useMemo(
    () => items.filter((i) => i.tipo === "camara"),
    [items]
  );

  // ── Búsqueda ─────────────────────────────────────────────────
  const itemsFiltrados = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return items;
    return items.reduce((acc, item) => {
      if (item.tipo === "sector") {
        const camarasFiltradas = item.camaras.filter((c) =>
          c.nombre.toLowerCase().includes(q)
        );
        if (item.nombre.toLowerCase().includes(q) || camarasFiltradas.length > 0) {
          acc.push({ ...item, camaras: item.nombre.toLowerCase().includes(q) ? item.camaras : camarasFiltradas });
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
    console.log("Guardar:", id);
  };

  const actualizarNombreSector = (id, valor) =>
    setItems((prev) =>
      prev.map((item) =>
        item.tipo === "sector" && item.id === id ? { ...item, nombre: valor } : item
      )
    );

  const actualizarNombreCamara = (sectorId, camaraId, valor) =>
    setItems((prev) =>
      prev.map((item) => {
        if (item.tipo === "camara" && item.id === camaraId) return { ...item, nombre: valor };
        if (item.tipo === "sector" && item.id === sectorId) {
          return { ...item, camaras: item.camaras.map((c) => c.id === camaraId ? { ...c, nombre: valor } : c) };
        }
        return item;
      })
    );

  // ── Confirmar creación desde el modal ────────────────────────
  const confirmarCreacion = async ({ tipo, nombre, hardwareId, camaraIds, sectorId }) => {
    if (tipo === "camara") {
      // Cámara nueva suelta desde hardware
      const response = await fetch('http://localhost:8000/camaras/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ usb_index: 0, nombre: nombre })
      })
      const data = await response.json()
      // data.id es el id real de CamaraConfig en la DB
      setItems((prev) => [...prev, { id: data.id, tipo: "camara", nombre }]);

    } else if (tipo === "sector") {
      // Sector nuevo: mueve las cámaras seleccionadas dentro del sector
      const nuevoId = Date.now();
      const camarasDelSector = items
        .filter((i) => i.tipo === "camara" && camaraIds.includes(i.id))
        .map(({ id, nombre }) => ({ id, nombre }));

      setItems((prev) => {
        const sinMovidas = prev.filter(
          (i) => !(i.tipo === "camara" && camaraIds.includes(i.id))
        );
        return [...sinMovidas, { id: nuevoId, tipo: "sector", nombre, camaras: camarasDelSector }];
      });
      console.log("Sector creado:", nombre, "con cámaras:", camaraIds);

    } else if (tipo === "agregarASector") {
      // Mueve cámaras sueltas al sector destino
      const camarasAMover = items
        .filter((i) => i.tipo === "camara" && camaraIds.includes(i.id))
        .map(({ id, nombre }) => ({ id, nombre }));

      setItems((prev) => {
        const sinMovidas = prev.filter(
          (i) => !(i.tipo === "camara" && camaraIds.includes(i.id))
        );
        return sinMovidas.map((item) =>
          item.tipo === "sector" && item.id === sectorId
            ? { ...item, camaras: [...item.camaras, ...camarasAMover] }
            : item
        );
      });
      console.log("Cámaras agregadas al sector:", sectorId, camaraIds);
    }
  };

  // ── Pinear ───────────────────────────────────────────────────
  const pinearCamara = (id) => {
    console.log("Pinear cámara:", id);
    // TODO: conectar con el grid
  };

  return {
    items: itemsFiltrados,
    camarasSueltas,
    query, setQuery,
    editandoId,
    toggleEdicion, guardarNombre,
    actualizarNombreSector, actualizarNombreCamara,
    confirmarCreacion,
    pinearCamara,
  };
}
