import { urlEvidencia } from "../VisorEvidencia/VisorEvidencia";
import "./EventoItem.css";

// { camara: "Deposito", tipo: "Paquete abandonado", fecha: "24/08/26 12:00" }
//
// 18/09: hasta hoy el historial estaba siempre vacio, asi que nadie vio una
// fila de verdad. Apenas empezo a cargar lo guardado se noto que las tres
// cosas en una linea no entran: el nombre de la camara quedaba en "cam-d..."
// y con "Paquete abandonado" desaparecia del todo. Van en dos lineas: arriba
// QUE paso, que es lo que uno busca al recorrer la lista; abajo donde y
// cuando.
//
// 26/09: si la alerta trajo evidencia, la miniatura de la captura va a la
// izquierda y un clic abre el clip. Sin evidencia la fila queda como antes.
export default function EventoItem({ camara, tipo, fecha, capturaUrl, clipUrl, revision, onVer }) {
  const miniatura = urlEvidencia(capturaUrl);
  const hayEvidencia = Boolean(capturaUrl || clipUrl);
  return (
    <li
      className={`evento-item ${hayEvidencia ? "evento-item--con-evidencia" : ""}`}
      onClick={hayEvidencia ? onVer : undefined}
      title={hayEvidencia ? "Ver lo que se vio" : undefined}
    >
      {hayEvidencia && (
        <span className="evento-item__miniatura">
          {miniatura
            ? <img src={miniatura} alt="" loading="lazy" />
            : <span className="evento-item__play">▶</span>}
          {clipUrl && <span className="evento-item__insignia">▶</span>}
        </span>
      )}
      <span className="evento-item__tipo">
        {tipo}
        {revision === "falsa" && <span className="evento-item__marca" title="Marcada como falsa alarma"> ✗</span>}
        {revision === "correcta" && <span className="evento-item__marca evento-item__marca--si" title="Confirmada"> ✓</span>}
      </span>
      <span className="evento-item__fecha">{fecha}</span>
      <span className="evento-item__camara" title={camara}>{camara}</span>
    </li>
  );
}
