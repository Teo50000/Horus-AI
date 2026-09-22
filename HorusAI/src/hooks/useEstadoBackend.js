import { useEffect, useRef, useState } from "react";
import { API_ESTADO } from "../config";

/**
 * useEstadoBackend
 *
 * Pregunta cada tanto que canales de aviso estan vivos del lado del backend.
 *
 * Hoy son dos: el websocket (por donde llegan las alertas al panel) y el mail
 * (por donde se avisa a los contactos de emergencia). El panel ya avisaba de
 * los dos que puede ver solo —websocket caido, modelos apagados—, pero el
 * mail no se ve desde aca: se manda del lado del servidor y si falla, falla
 * en una consola que nadie mira.
 *
 * 22/09, medido en la base: 41 alertas de severidad 2 guardadas, 5 eventos,
 * CERO mails. No existia el archivo .env con las credenciales, asi que todos
 * los envios morian en el login. La alerta se veia igual de bien en el panel.
 *
 * `mail` queda en null mientras no se sepa. null NO es "anda": es "todavia no
 * pregunte", y el cartel lo trata distinto a proposito.
 */
export function useEstadoBackend(cadaMs = 15000) {
  const [mail, setMail] = useState(null);   // {ok, motivo} | null
  const vivo = useRef(true);

  useEffect(() => {
    vivo.current = true;

    const preguntar = async () => {
      try {
        const r = await fetch(API_ESTADO, { cache: "no-store" });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const e = await r.json();
        if (!vivo.current) return;
        setMail(e.mail ?? null);
      } catch {
        // Backend caido: de eso ya avisa el cartel del websocket. Volvemos a
        // "no se", que no es lo mismo que "el mail anda".
        if (!vivo.current) return;
        setMail(null);
      }
    };

    preguntar();
    const id = setInterval(preguntar, cadaMs);
    return () => { vivo.current = false; clearInterval(id); };
  }, [cadaMs]);

  return { mail };
}
