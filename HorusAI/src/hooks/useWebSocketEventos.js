/*import { useEffect, useRef, useState } from "react";

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
/*export function useWebSocketEventos(url) {
  const [eventos, setEventos] = useState([]);
  const [conectado, setConectado] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
      setConectado(true);
      console.log("WebSocket conectado:", url);
    };

    ws.onmessage = (event) => {
      try {
        const nuevoEvento = JSON.parse(event.data);
        // Agrega el evento nuevo al principio de la lista
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
}*/


/*type CamaraConfig = {
  id: number;
  nombre: string;
  rtsp_url: string;
};

type Alerta = {
  camera_id: number;
  event_type: string;
  confidence: number;
  timestamp: string;
  nombre_camara: string;
  description: string;
};*/

// useWebSocketEventos.js
import { useEffect, useState, useRef } from "react";
import WebSocket from "@tauri-apps/plugin-websocket";

export default function App() {
  const [mensaje, setMensaje] = useState("");
  const [historial, setHistorial] = useState([]);
  const [conectado, setConectado] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let unlisten;

    async function iniciarWS() {
      try {
        // Conexión local al servidor Python
        const ws = await WebSocket.connect("ws://127.0.0.1:8000");
        wsRef.current = ws;
        setConectado(true);

        // Escuchar mensajes del servidor Python
        unlisten = await ws.addListener((msg) => {
          // El plugin de Tauri envía un objeto con la propiedad 'data'
          if (msg.type === "Text") {
            setHistorial((prev) => [...prev, `Servidor: ${msg.data}`]);
          }
        });
      } catch (error) {
        console.error("Error al conectar WebSocket:", error);
      }
    }

    iniciarWS();

    // Limpieza al desmontar el componente de React
    return () => {
      if (unlisten) unlisten();
      if (wsRef.current) wsRef.current.disconnect();
    };
  }, []);

  const enviarMensaje = async () => {
    if (wsRef.current && mensaje.trim() !== "") {
      await wsRef.current.send(mensaje);
      setHistorial((prev) => [...prev, `Tú: ${mensaje}`]);
      setMensaje("");
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h3>Estado: {conectado ? "🟢 Conectado" : "🔴 Desconectado"}</h3>
      
      <div style={{ border: "1px solid #ccc", height: "200px", overflowY: "scroll", padding: "10px" }}>
        {historial.map((txt, i) => <p key={i}>{txt}</p>)}
      </div>

      <input 
        value={mensaje} 
        onChange={(e) => setMensaje(e.target.value)} 
        placeholder="Escribe a Python..." 
      />
      <button onClick={enviarMensaje}>Enviar</button>
    </div>
  );
}

