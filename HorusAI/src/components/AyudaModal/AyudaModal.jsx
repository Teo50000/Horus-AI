import { useEffect, useRef } from "react";
import "./AyudaModal.css";

/**
 * AyudaModal — guía corta de cómo usar el panel. Se abre con el "?" del
 * sidebar y se cierra con Esc, con el botón o tocando afuera.
 */
export default function AyudaModal({ abierto, onClose }) {
  const cerrarRef = useRef(null);

  useEffect(() => {
    if (!abierto) return undefined;
    cerrarRef.current?.focus();
    const alTeclear = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", alTeclear);
    return () => window.removeEventListener("keydown", alTeclear);
  }, [abierto, onClose]);

  if (!abierto) return null;

  return (
    <div className="ayuda-modal__fondo" onMouseDown={onClose}>
      <div
        className="ayuda-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="ayuda-titulo"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="ayuda-modal__cabecera">
          <h2 id="ayuda-titulo" className="ayuda-modal__titulo">Cómo usar Horus</h2>
          <button ref={cerrarRef} type="button" className="ayuda-modal__cerrar"
                  onClick={onClose} aria-label="Cerrar ayuda">✕</button>
        </header>

        <div className="ayuda-modal__cuerpo">
          <section>
            <h3>Ver cámaras</h3>
            <ul>
              <li>Abrí <b>Cámaras</b> en la barra de la izquierda (o tocá un espacio vacío de la grilla).</li>
              <li>Con el <b>pin</b> de cada cámara la sumás a la grilla. El <b>ojo</b> la previsualiza sin sumarla.</li>
              <li>Pasá el mouse por una cámara de la grilla y tocá <b>✕</b> para sacarla. La grilla se recuerda entre sesiones.</li>
            </ul>
          </section>

          <section>
            <h3>Qué significan los recuadros</h3>
            <ul className="ayuda-modal__leyenda">
              <li><span className="ayuda-modal__muestra" style={{ borderColor: "#5adc5a" }} />Persona detectada</li>
              <li><span className="ayuda-modal__muestra" style={{ borderColor: "#ff3c3c" }} />Arma (pistola o cuchillo)</li>
              <li><span className="ayuda-modal__muestra" style={{ borderColor: "#ff8c00" }} />Fuego o llama</li>
              <li><span className="ayuda-modal__muestra ayuda-modal__muestra--fina" style={{ borderColor: "#ffa03c" }} />Algo que se dejó de ver hace un instante</li>
            </ul>
          </section>

          <section>
            <h3>Estado del sistema</h3>
            <p>Abajo a la izquierda hay tres indicadores. En verde todo anda; si uno se pone
              naranja o rojo y late, pasá el mouse por encima para ver qué pasa.</p>
            <ul>
              <li><b>En vivo</b>: el panel recibe las alertas.</li>
              <li><b>IA</b>: cuántas cámaras están analizando los modelos.</li>
              <li><b>Mail</b>: si las alertas también salen por mail.</li>
            </ul>
          </section>

          <section>
            <h3>Alertas</h3>
            <p>Cuando los modelos detectan algo aparece un aviso en pantalla (se puede apagar en
              <b> Ajustes</b>). Todo queda en <b>Historial</b>, con la captura y el clip del momento.</p>
          </section>
        </div>
      </div>
    </div>
  );
}
