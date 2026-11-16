import { useState } from "react";
import "./InputContrasena.css";

/**
 * Campo de contraseña con botón para verla.
 *
 * 16/11: el ojito que se veía antes era el botón NATIVO de Edge (la app de
 * escritorio corre sobre WebView2): en Chrome no aparece y en Edge desaparece
 * apenas el campo pierde el foco. Este es propio, anda igual en todos lados y
 * se puede usar con el teclado. El nativo se oculta para que no haya dos.
 *
 * Acepta las mismas props que un <input>; `className` va al input.
 */
export default function InputContrasena({ className = "", ...props }) {
  const [visible, setVisible] = useState(false);

  return (
    <span className="input-contrasena">
      <input
        {...props}
        type={visible ? "text" : "password"}
        className={`${className} input-contrasena__campo`}
        spellCheck={false}
      />
      <button
        type="button"
        className="input-contrasena__ojo"
        onClick={() => setVisible((v) => !v)}
        aria-label={visible ? "Ocultar contraseña" : "Mostrar contraseña"}
        aria-pressed={visible}
        title={visible ? "Ocultar contraseña" : "Mostrar contraseña"}
      >
        {visible ? (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
               strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M3 3l18 18" />
            <path d="M10.6 5.1A10.4 10.4 0 0 1 12 5c5 0 9 4.5 10 7-0.4 1-1.2 2.4-2.4 3.7" />
            <path d="M6.6 6.6C4.4 8 2.8 10.2 2 12c1 2.5 5 7 10 7 1.9 0 3.6-0.6 5.1-1.5" />
            <path d="M9.9 9.9a3 3 0 0 0 4.2 4.2" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"
               strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M2 12c1-2.5 5-7 10-7s9 4.5 10 7c-1 2.5-5 7-10 7S3 14.5 2 12z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        )}
      </button>
    </span>
  );
}
