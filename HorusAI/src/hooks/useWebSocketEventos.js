import { useEffect, useRef, useState } from "react";

function mapearTipo(eventType) {
  const mapa = {
    fire: "Incendio",
    assault: "Agresión",
    faint: "Desmayo",
    desmayo: "Desmayo",
    incendio: "Incendio",
  };
  return mapa[eventType?.toLowerCase()] ?? eventType;
}

export function useWebSocketEventos(url) {
  const [eventos, setEventos]   = useState([]);
  const [conectado, setConectado] = useState(false);
  const wsRef        = useRef(null);
  const retryTimeout = useRef(null);
  const intentos     = useRef(0);
  const MAX_INTENTOS = 10;   // deja de intentar después de 10 fallos seguidos
  const DELAY_BASE   = 2000; // empieza con 2 segundos, va subiendo

  useEffect(() => {
    let cancelado = false; // evita reconectar si el componente se desmontó

    const conectar = () => {
      if (cancelado) return;

      console.log(`WebSocket: intentando conectar (intento ${intentos.current + 1})...`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelado) { ws.close(); return; }
        console.log("WebSocket conectado:", url);
        setConectado(true);
        intentos.current = 0; // resetear contador al conectar exitosamente
      };

      ws.onmessage = (event) => {
        try {
          const raw = JSON.parse(event.data);
          const eventoMapeado = {
            id:         `${raw.camera_id}-${raw.timestamp}`,
            tipo:       mapearTipo(raw.event_type),
            camara:     raw.nombre_camara ?? `Camara ${raw.camera_id}`,
            fecha:      raw.timestamp,
            confidence: raw.confidence,
          };
          setEventos((prev) => [eventoMapeado, ...prev]);
        } catch (err) {
          console.error("Error al parsear evento:", err);
        }
      };

      ws.onclose = () => {
        if (cancelado) return;
        setConectado(false);
        console.log("WebSocket desconectado");

        // Reintentar con backoff exponencial
        if (intentos.current < MAX_INTENTOS) {
          const delay = Math.min(DELAY_BASE * 2 ** intentos.current, 30000); // máximo 30s
          console.log(`WebSocket: reintentando en ${delay / 1000}s...`);
          intentos.current += 1;
          retryTimeout.current = setTimeout(conectar, delay);
        } else {
          console.warn("WebSocket: máximo de intentos alcanzado, dejando de reintentar.");
        }
      };

      ws.onerror = () => {
        // onerror siempre va seguido de onclose, así que la reconexión
        // se maneja ahí — solo logueamos
        console.warn("WebSocket: error de conexión");
      };
    };

    conectar();

    return () => {
      cancelado = true;
      clearTimeout(retryTimeout.current);
      wsRef.current?.close();
    };
  }, [url]);

  return { eventos, conectado };
}
