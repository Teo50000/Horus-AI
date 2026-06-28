import { useState, useCallback } from "react";

// Cada slot puede tener:
// null → vacío
// { tipo: "camara", id, nombre }
// { tipo: "sector", nombre, camaras: [{id, nombre}], indice: 0 }

const TOTAL_SLOTS = 8;
const slots_iniciales = Array(TOTAL_SLOTS).fill(null);

export function useGrid() {
  const [slots, setSlots] = useState(slots_iniciales);

  // Encuentra el primer slot vacío, o null si están todos ocupados
  const primerSlotVacio = (lista) => lista.findIndex((s) => s === null);

  const pinearCamara = useCallback((camara) => {
    setSlots((prev) => {
      // Si la cámara ya está pineada, no hacer nada
      if (prev.some((s) => s?.tipo === "camara" && s.id === camara.id)) return prev;
      const idx = primerSlotVacio(prev);
      if (idx === -1) return prev; // lleno, ignorar
      const next = [...prev];
      next[idx] = { tipo: "camara", id: camara.id, nombre: camara.nombre };
      return next;
    });
  }, []);

  const pinearSector = useCallback((sector) => {
    setSlots((prev) => {
      // Si el sector ya está pineado, no hacer nada
      if (prev.some((s) => s?.tipo === "sector" && s.nombre === sector.nombre)) return prev;
      const idx = primerSlotVacio(prev);
      if (idx === -1) return prev;
      const next = [...prev];
      next[idx] = {
        tipo: "sector",
        nombre: sector.nombre,
        camaras: sector.camaras,
        indice: 0,
      };
      return next;
    });
  }, []);

  // Navegar entre cámaras de un sector pineado
  const navegarSector = useCallback((slotIdx, direccion) => {
    setSlots((prev) => {
      const slot = prev[slotIdx];
      if (!slot || slot.tipo !== "sector") return prev;
      const total = slot.camaras.length;
      const nuevoIndice =
        direccion === "siguiente"
          ? (slot.indice + 1) % total
          : (slot.indice - 1 + total) % total;
      const next = [...prev];
      next[slotIdx] = { ...slot, indice: nuevoIndice };
      return next;
    });
  }, []);

  const vaciarSlot = useCallback((slotIdx) => {
    setSlots((prev) => {
      const next = [...prev];
      next[slotIdx] = null;
      return next;
    });
  }, []);

  return { slots, pinearCamara, pinearSector, navegarSector, vaciarSlot };
}
