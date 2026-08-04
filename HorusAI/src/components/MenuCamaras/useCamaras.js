import { useState, useEffect, useMemo } from "react";

const API = "http://localhost:8000/camaras";

function dbAItem(config) {
  return {
    id: config.id,
    tipo: "camara",
    nombre: config.nombre ?? `Camara ${config.id}`,
    usb_index: config.usb_index,
  };
}

export function useCamaras() {
  const [items, setItems]       = useState([]);
  const [cargando, setCargando] = useState(true);
  const [query, setQuery]       = useState("");
  const [editandoId, setEditandoId] = useState(null);

  // ── Carga inicial desde la DB ────────────────────────────────
  useEffect(() => {
    fetch(`${API}/config`)
      .then((res) => {
        if (res.status === 404) return [];
        if (!res.ok) throw new Error(`Error ${res.status}`);
        return res.json();
      })
      .then((configs) => setItems(configs.map(dbAItem)))
      .catch((err) => console.error("Error al cargar cámaras:", err))
      .finally(() => setCargando(false));
  }, []);

  // ── Cámaras sueltas ──────────────────────────────────────────
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
        const sectorCoincide = item.nombre.toLowerCase().includes(q);
        if (sectorCoincide || camarasFiltradas.length > 0) {
          acc.push({ ...item, camaras: sectorCoincide ? item.camaras : camarasFiltradas });
        }
      } else {
        if (item.nombre.toLowerCase().includes(q)) acc.push(item);
      }
      return acc;
    }, []);
  }, [items, query]);

  // ── Edición ──────────────────────────────────────────────────
  const toggleEdicion = (id) =>
    setEditandoId((prev) => (prev === id ? null : id));

  const guardarNombre = async (idEdicion) => {
    setEditandoId(null);
    const esSector   = idEdicion.startsWith("s-");
    const idNumerico = parseInt(idEdicion.split("-")[1]);

    if (esSector) return; // sectores solo en frontend por ahora

    const camara = items
      .flatMap((i) => (i.tipo === "sector" ? i.camaras : [i]))
      .find((c) => c.id === idNumerico);

    if (!camara) return;

    try {
      await fetch(`${API}/config/${idNumerico}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nombre: camara.nombre }),
      });
    } catch (err) {
      console.error("Error al guardar nombre:", err);
    }
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
        if (item.tipo === "camara" && item.id === camaraId)
          return { ...item, nombre: valor };
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
  const confirmarCreacion = async ({ tipo, nombre, hardwareId, camaraIds, sectorId }) => {
    if (tipo === "camara") {
      try {
        const res = await fetch(`${API}/config`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ usb_index: hardwareId, nombre }),
        });
        const data = await res.json();
        setItems((prev) => [...prev, dbAItem(data)]);
      } catch (err) {
        console.error("Error al crear cámara:", err);
      }

    } else if (tipo === "sector") {
      const nuevoId = `sector-${Date.now()}`;
      const camarasDelSector = items
        .filter((i) => i.tipo === "camara" && camaraIds.includes(i.id))
        .map(({ id, nombre }) => ({ id, nombre }));

      setItems((prev) => {
        const sinMovidas = prev.filter(
          (i) => !(i.tipo === "camara" && camaraIds.includes(i.id))
        );
        return [...sinMovidas, { id: nuevoId, tipo: "sector", nombre, camaras: camarasDelSector }];
      });

    } else if (tipo === "agregarASector") {
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
    }
  };

  return {
    items: itemsFiltrados,
    cargando,
    camarasSueltas,
    query, setQuery,
    editandoId,
    toggleEdicion, guardarNombre,
    actualizarNombreSector, actualizarNombreCamara,
    confirmarCreacion,
  };
}
