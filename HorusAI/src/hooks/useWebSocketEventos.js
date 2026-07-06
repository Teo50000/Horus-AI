import { useEffect, useRef, useState } from "react";

/**
 * useWebSocketEventos
 * Se conecta al WebSocket de FastAPI y mantiene un array de eventos
 * que crece en tiempo real a medida que el backend los emite.
 *
 * No reemplaza useHistorial: le da el array `eventos` que necesita.
 *
 * Uso:
 *   const { eventos, conectado } = useWebSocketEventos("ws://localhost:8000/ws/eventos");
 *   const historial = useHistorial(eventos);
 */
export function useWebSocketEventos(url) {
  const [eventos, setEventos] = useState([]);
  const [conectado, setConectado] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConectado(true);
      console.log("WebSocket conectado:", url);
    };

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

    ws.onmessage = (event) => {
      try {
        console.log("RAW mensaje recibido:", event.data);
        const nuevoEvento = JSON.parse(event.data);
        // Agrega el evento nuevo al principio de la lista
        const eventoMapeado = {
          tipo: mapearTipo(nuevoEvento.event_type),
          camara: nuevoEvento.camera_id,
          fecha: nuevoEvento.timestamp,
          confidence: nuevoEvento.confidence,
          detalle: nuevoEvento.description,
        }
        setEventos((prev) => [nuevoEvento, ...prev]);
      } catch (err) {
        console.error("Error al parsear evento:", err);
      }
    };

    ws.onclose = () => {
      setConectado(false);
      console.log("WebSocket desconectado");
    };

    ws.onerror = (err) => {
      console.error("Error en WebSocket:", err);
    };

    // Limpieza: cerrar conexión al desmontar el componente
    return () => {
      ws.close();
    };
  }, [url]);

  return { eventos, conectado };
}

