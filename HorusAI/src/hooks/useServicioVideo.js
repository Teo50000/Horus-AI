import { useEffect, useRef, useState } from "react";
import { SERVICIO_URL } from "../config";

/**
 * useServicioVideo
 *
 * Pregunta cada tanto si el servicio de modelos esta corriendo y que camaras
 * tiene abiertas.
 *
 * Por que existe: en Windows una webcam la abre UN proceso a la vez. El panel
 * le pedia el video al backend, el backend abria la camara, y el servicio se
 * quedaba sin poder abrirla — los modelos corriendo sobre nada. El sintoma es
 * "me detecta la camara pero el modelo no corre cuando la cam esta prendida".
 *
 * La camara la abre el servicio y nadie mas. El servicio ya publica el video
 * con las cajas dibujadas en su propio puerto, asi que el panel muestra ese,
 * que ademas es el que uno quiere ver. Si el servicio no esta, el panel se cae
 * solo al video crudo del backend.
 */
export function useServicioVideo(cadaMs = 4000) {
  const [activo, setActivo]   = useState(false);
  const [fase, setFase]       = useState(null); // "cargando" | "listo" | null
  const [camaras, setCamaras] = useState({});   // config_id -> {camara, estado, error}
  // 16/11: true despues de la PRIMERA respuesta (o falla). Sin esto la celda
  // arrancaba pidiendo el video al backend (activo=false al montar) y a los
  // 4 s lo volvia a pedir al servicio: dos conexiones y un parpadeo.
  const [consultado, setConsultado] = useState(false);
  // 16/11: sube cada vez que el servicio (re)arranca. Un <img> MJPEG al que
  // le cortan la conexión NO dispara onError: se queda congelado en el
  // último cuadro, en silencio. La celda usa este número en la URL, así que
  // cuando el servicio vuelve, el video se reconecta solo.
  const [generacion, setGeneracion] = useState(0);
  const arranque = useRef(null);
  const vivo = useRef(true);

  useEffect(() => {
    vivo.current = true;

    const preguntar = async () => {
      try {
        const r = await fetch(`${SERVICIO_URL}/estado`, { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const e = await r.json();
        if (!vivo.current) return;
        const porId = {};
        for (const c of e.camaras ?? []) {
          if (c.config_id !== null && c.config_id !== undefined) {
            porId[c.config_id] = c;
          }
        }
        setCamaras(porId);
        setFase(e.fase ?? "listo");
        setActivo(true);
        setConsultado(true);
        // Hora (aprox.) en que arrancó el servicio. Si se corre más de 5 s,
        // es otro proceso: reconectar los videos.
        const inicio = Date.now() - (Number(e.arriba_s) || 0) * 1000;
        if (arranque.current !== null && Math.abs(inicio - arranque.current) > 5000) {
          setGeneracion((g) => g + 1);
        }
        arranque.current = inicio;
      } catch {
        // Que el servicio no este es normal: se puede usar el panel solo para
        // ver camaras. No es un error, es otro modo.
        if (!vivo.current) return;
        setActivo(false);
        setFase(null);
        setCamaras({});
        setConsultado(true);
        if (arranque.current !== null) {
          arranque.current = null;
          setGeneracion((g) => g + 1);
        }
      }
    };

    preguntar();
    const id = setInterval(preguntar, cadaMs);
    return () => { vivo.current = false; clearInterval(id); };
  }, [cadaMs]);

  /** La URL de la que hay que sacar el video de esta camara. */
  const urlDe = (configId) => {
    const c = camaras[configId];
    if (activo && c) {
      return {
        url: `${SERVICIO_URL}/camaras/${c.camara}/stream`,
        conIA: true,
        estado: c.estado,
        error: c.error,
      };
    }
    return { url: null, conIA: false, estado: null, error: null };
  };

  return { activo, fase, camaras, consultado, generacion, urlDe };
}
