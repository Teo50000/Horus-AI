import { useState, useEffect, useRef, useCallback } from "react";

/**
 * useAlertaActiva
 * Mira el evento más reciente que llega por WebSocket y decide si hay que
 * mostrar el AlertModal.
 *
 *  - Ignora los eventos "normal" (no son una alerta).
 *  - Respeta el toggle "Alerta de eventos en pantalla" de Ajustes (habilitada).
 *  - Si llega otra alerta con el modal abierto, la reemplaza por la más nueva.
 *
 * Uso:
 *   const { alerta, cerrarAlerta } = useAlertaActiva(eventos, ajustes.alertaEnPantalla);
 */
export function useAlertaActiva(eventos, habilitada = true) {
  const [alerta, setAlerta] = useState(null);

  // Último evento ya procesado: evita que al reactivar el toggle
  // reaparezca una alerta vieja.
  const ultimoIdVisto = useRef(null);

  useEffect(() => {
    const ultimo = eventos[0];
    if (!ultimo || ultimo.id === ultimoIdVisto.current) return;

    ultimoIdVisto.current = ultimo.id;

    if (!habilitada) return;
    if (ultimo.tipo?.toLowerCase() === "normal") return;

    setAlerta(ultimo);
  }, [eventos, habilitada]);

  const cerrarAlerta = useCallback(() => setAlerta(null), []);

  return { alerta, cerrarAlerta };
}
