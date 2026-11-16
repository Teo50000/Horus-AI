import { useState, useCallback, useEffect } from "react";

// Cada slot puede tener:
// null → vacío
// { tipo: "camara", id, nombre }
// { tipo: "sector", nombre, camaras: [{id, nombre}], indice: 0 }

const TOTAL_SLOTS = 8;
const slots_iniciales = Array(TOTAL_SLOTS).fill(null);

// 16/11: la grilla se perdía en cada arranque y había que volver a pinear
// todas las cámaras. Se recuerda en este equipo (localStorage). Si el
// almacenamiento no está disponible o lo guardado no tiene la forma
// esperada, se arranca vacía como antes.
const CLAVE = "horus.grilla.v1";

function slotValido(s) {
  if (s === null) return true;
  if (!s || typeof s !== "object") return false;
  if (s.tipo === "camara") return s.id !== undefined && typeof s.nombre === "string";
  if (s.tipo === "sector") {
    return typeof s.nombre === "string" && Array.isArray(s.camaras)
      && s.camaras.length > 0 && Number.isInteger(s.indice);
  }
  return false;
}

function leerGuardado() {
  try {
    const crudo = window.localStorage.getItem(CLAVE);
    if (!crudo) return slots_iniciales;
    const lista = JSON.parse(crudo);
    if (!Array.isArray(lista) || lista.length !== TOTAL_SLOTS) return slots_iniciales;
    return lista.map((s) => (slotValido(s) ? s : null)).map((s) =>
      s?.tipo === "sector" ? { ...s, indice: Math.min(s.indice, s.camaras.length - 1) } : s);
  } catch {
    return slots_iniciales;
  }
}

export function useGrid() {
  const [slots, setSlots] = useState(leerGuardado);

  useEffect(() => {
    try {
      window.localStorage.setItem(CLAVE, JSON.stringify(slots));
    } catch {
      // sin almacenamiento (modo privado, cuota): la grilla igual anda
    }
  }, [slots]);

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
  // 26/09: estas dos funciones hacian fetch por su cuenta, y mal:
  //  - `POST /video/stop_feed` SIN id, una ruta que no existe (405);
  //  - `GET /video/video_feed/${nuevoIndice}`: el INDICE del carrusel en vez
  //    del id de la camara, y con un fetch cuyo cuerpo nadie lee. Eso abria
  //    un stream que quedaba colgado en el backend, con la camara tomada, y
  //    el servicio de modelos sin poder abrirla. La celda ya pide su video
  //    sola (CeldaCamara) y ya lo corta al quitarla: aca no hace falta nada.
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
