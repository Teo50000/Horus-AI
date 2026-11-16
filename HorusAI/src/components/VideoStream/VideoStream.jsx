import { useEffect, useRef, useState } from "react";
import "./VideoStream.css";

// Esperas entre reintentos: rápido al principio (el servicio suele tardar un
// par de segundos en enganchar una cámara recién dada de alta) y después más
// espaciado para no martillar a un backend caído.
const ESPERAS_MS = [800, 1500, 3000, 5000, 8000];

/**
 * VideoStream — un MJPEG con skeleton mientras carga y reconexión sola.
 *
 * 16/11 — tres problemas del componente anterior:
 *  1. Un solo error (un 503 porque el servicio todavía no enganchó la cámara,
 *     un corte de red) dejaba "Sin señal" PARA SIEMPRE: no reintentaba.
 *  2. El spinner quedaba detrás del video y "Conectando..." se veía en cada
 *     borde transparente; no había forma de saber cuándo llegó el primer
 *     cuadro.
 *  3. Al desmontar, la conexión MJPEG podía quedar abierta un rato.
 *
 * Estados: "cargando" (skeleton) → "ok" (video) → si se corta, "reconectando"
 * (el último cuadro queda congelado y atenuado, con un aviso) y vuelve solo.
 */
export default function VideoStream({ src, alt = "", aviso = null }) {
  const [fase, setFase]       = useState("cargando"); // cargando | ok | reconectando
  const [intento, setIntento] = useState(0);
  const imgRef = useRef(null);
  const reintento = useRef(null);

  // Fuente nueva: volver a empezar.
  useEffect(() => {
    setFase("cargando");
    setIntento(0);
  }, [src]);

  // El evento `load` de un MJPEG no es confiable en todos los WebView: se
  // mira directamente si ya hay un cuadro decodificado.
  useEffect(() => {
    if (!src || fase === "ok") return undefined;
    const id = setInterval(() => {
      const img = imgRef.current;
      if (img && img.naturalWidth > 0 && img.complete !== false) {
        setFase("ok");
      }
    }, 200);
    return () => clearInterval(id);
  }, [src, fase, intento]);

  useEffect(() => () => {
    clearTimeout(reintento.current);
    // Cortar la conexión MJPEG al desmontar (si no, el navegador puede
    // dejarla abierta y el servicio sigue codificando para nadie).
    if (imgRef.current) imgRef.current.src = "data:,";
  }, []);

  const alFallar = () => {
    setFase((f) => (f === "ok" || f === "reconectando" ? "reconectando" : "cargando"));
    clearTimeout(reintento.current);
    const espera = ESPERAS_MS[Math.min(intento, ESPERAS_MS.length - 1)];
    reintento.current = setTimeout(() => setIntento((n) => n + 1), espera);
  };

  const url = src
    ? `${src}${src.includes("?") ? "&" : "?"}r=${intento}`
    : null;

  return (
    <div className={`video-stream video-stream--${fase}`}>
      {fase !== "ok" && fase !== "reconectando" && (
        <div className="video-stream__skeleton skeleton" aria-hidden="true">
          <svg className="video-stream__icono" viewBox="0 0 24 24" fill="none"
               stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"
               strokeLinejoin="round">
            <path d="M15 10l4.55-2.28A1 1 0 0 1 21 8.62v6.76a1 1 0 0 1-1.45.9L15 14" />
            <rect x="3" y="6" width="12" height="12" rx="2" />
          </svg>
          <span className="video-stream__texto">
            {intento === 0 ? "Conectando cámara…" : "Esperando video…"}
          </span>
        </div>
      )}

      {url && (
        <img
          ref={imgRef}
          src={url}
          alt={alt}
          className="video-stream__img"
          draggable={false}
          onLoad={() => setFase("ok")}
          onError={alFallar}
        />
      )}

      {(fase === "reconectando" || aviso) && (
        <div className="video-stream__aviso" role="status">
          <span className="video-stream__aviso-punto" aria-hidden="true" />
          {aviso ?? "Reconectando…"}
        </div>
      )}
    </div>
  );
}
