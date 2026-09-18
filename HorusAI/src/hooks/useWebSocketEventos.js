import { useEffect, useRef, useState } from "react";

// Los tipos que emite de verdad la capa de fusión (horus/06_fusion_decision).
// 18/09: este mapa conocía 5 claves —fire, assault, faint, desmayo, incendio—
// y la fusión emite 10. Solo coincidía "incendio": el resto caía al `??` y se
// mostraba crudo, así que en el panel convivían "Incendio" y "paquete_abandonado".
// Las claves en inglés eran de una versión anterior y ya no las emite nadie; se
// dejan porque no molestan y porque puede haber filas viejas en la base.
const TIPOS = {
  incendio: "Incendio",
  arma: "Arma",
  pelea: "Pelea",
  agresion: "Agresión",
  caida: "Caída",
  intrusion: "Intrusión",
  merodeo: "Merodeo",
  paquete_abandonado: "Paquete abandonado",
  robo: "Robo",
  accion_externa: "Acción externa",
  // histórico
  fire: "Incendio",
  assault: "Agresión",
  faint: "Desmayo",
  desmayo: "Desmayo",
};

export function mapearTipo(eventType) {
  return TIPOS[eventType?.toLowerCase()] ?? eventType;
}

// Los tipos que el panel puede ofrecer como filtro, ya en castellano y sin
// repetir: "Incendio" sale tanto de `incendio` como de `fire`.
//
// Antes esta lista estaba escrita a mano en useHistorial.js y tenia tres
// entradas —Desmayo, Incendio, Agresion— de los diez que emite la fusion. Si
// una caida no se podia filtrar, no era un bug visible: simplemente el boton
// no existia. Derivandola de TIPOS, agregar una regla nueva en la fusion
// alcanza para que aparezca el filtro.
export const TIPOS_EN_ESPANOL = [...new Set(Object.values(TIPOS))].sort(
  (a, b) => a.localeCompare(b, "es")
);

// Los plurales, escritos a mano y a proposito.
//
// El boton de filtro mostraba `tipo + "s"`, que con tres tipos cortitos
// —Desmayos, Incendios, Agresións— ya estaba mal en uno y nadie lo noto. Con
// diez queda "Intrusións", "Acción externas" y "Paquete abandonados". El
// castellano pluraliza con -es tras consonante, pierde la tilde en -ion ->
// -iones, y en "paquete abandonado" hay que pluralizar las DOS palabras. No
// hay regla corta que acierte: se escriben.
export const PLURALES = {
  "Incendio":           "Incendios",
  "Arma":               "Armas",
  "Pelea":              "Peleas",
  "Agresión":           "Agresiones",
  "Caída":              "Caídas",
  "Intrusión":          "Intrusiones",
  "Merodeo":            "Merodeos",
  "Paquete abandonado": "Paquetes abandonados",
  "Robo":               "Robos",
  "Acción externa":     "Acciones externas",
  "Desmayo":            "Desmayos",
};

export function enPlural(tipo) {
  return PLURALES[tipo] ?? tipo;
}

// La fecha como se lee, no como la manda el backend.
//
// Horus manda ISO con huso: "2026-08-24T12:00:11.000+00:00". Mientras el
// historial estuvo vacio nadie lo vio; apenas empezo a cargar lo guardado, cada
// fila paso a ser una tira de 29 caracteres que tapa el nombre de la camara.
// Se muestra dd/mm/aa hh:mm, que ademas es lo que alguien escribe en el
// buscador. Si no se puede parsear se devuelve tal cual: mejor un ISO feo que
// un "Invalid Date".
export function fechaCorta(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso);
  const dd = String(d.getDate()).padStart(2, "0");
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const aa = String(d.getFullYear()).slice(-2);
  const hh = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  return `${dd}/${mm}/${aa} ${hh}:${mi}`;
}

// UN solo lugar donde se decide que forma tiene un evento para el panel.
//
// Hay dos fuentes: el websocket (en vivo) y GET /alertas (lo guardado). Si
// cada una arma el objeto por su cuenta, tarde o temprano difieren en un
// campo y el historial muestra algo distinto a lo que se vio en vivo.
export function eventoDelSocket(raw) {
  return {
    id:         raw.alerta?.id ?? `${raw.camera_id}-${raw.timestamp}`,
    tipo:       mapearTipo(raw.event_type),
    tipoCrudo:  raw.event_type,
    camara:     raw.nombre_camara ?? `Camara ${raw.camera_id}`,
    fecha:      fechaCorta(raw.timestamp),
    fechaIso:   raw.timestamp,
    confidence: raw.confidence,
    severidad:  raw.alerta?.severidad ?? null,
    envivo:     true,
  };
}

// Una fila de GET /alertas, con los nombres que usa la base.
export function eventoDeLaBase(fila) {
  return {
    id:         fila.evento_id,
    tipo:       mapearTipo(fila.tipo),
    tipoCrudo:  fila.tipo,
    camara:     fila.nombre_camara ?? fila.camara ?? "?",
    fecha:      fechaCorta(fila.ts_inicio),
    fechaIso:   fila.ts_inicio,
    confidence: fila.confianza,
    severidad:  fila.severidad ?? null,
    envivo:     false,
  };
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
          setEventos((prev) => [eventoDelSocket(raw), ...prev]);
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
