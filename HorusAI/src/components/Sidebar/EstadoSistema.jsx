import "./EstadoSistema.css";

/**
 * EstadoSistema — los tres canales de los que depende que una alerta llegue.
 *
 * 16/11: antes eran tres carteles fijos arriba a la derecha ("En vivo",
 * "Analizando 2", "Alertas sin mail") encima de la cámara de esa esquina. La
 * información importa —"no hubo alertas" y "nadie las estaba escuchando" no
 * pueden verse igual (ver los comentarios de 18/09 y 22/09)— pero no puede
 * tapar video. Ahora viven en el sidebar, que es fijo y no tapa nada:
 *
 *  - En calma son tres puntos discretos con una palabra.
 *  - Si algo falla, el punto se pone naranja/rojo y late, así sigue siendo
 *    imposible no notarlo.
 *  - El detalle (qué significa y qué hacer) aparece al pasar el mouse o al
 *    enfocarlo con el teclado.
 */
function estadoPanel(conectado) {
  return conectado
    ? { nivel: "ok", corto: "En vivo", titulo: "Recibiendo alertas",
        detalle: "El panel está conectado al backend y recibe las alertas en tiempo real." }
    : { nivel: "error", corto: "Offline", titulo: "Sin conexión al backend",
        detalle: "El panel no está recibiendo alertas. Si pasa algo, no se va a ver acá. Revisá que el backend esté corriendo." };
}

function estadoModelos(servicio) {
  const n = Object.keys(servicio?.camaras ?? {}).length;
  if (!servicio?.consultado) {
    return { nivel: "nose", corto: "IA…", titulo: "Consultando los modelos", detalle: "Todavía no tengo respuesta del servicio de modelos." };
  }
  if (!servicio.activo) {
    return { nivel: "aviso", corto: "IA off", titulo: "Modelos apagados",
      detalle: "El servicio de modelos no está corriendo: el video se ve, pero nadie lo está analizando." };
  }
  if (servicio.fase === "cargando") {
    return { nivel: "cargando", corto: "IA…", titulo: "Cargando modelos",
      detalle: "Subiendo los modelos a memoria. Tarda entre 15 y 40 segundos." };
  }
  if (n === 0) {
    return { nivel: "aviso", corto: "IA · 0", titulo: "Modelos sin cámaras",
      detalle: "Los modelos están cargados pero no les llega ninguna cámara. Dá de alta una cámara o revisá el backend." };
  }
  return { nivel: "ok", corto: `IA · ${n}`, titulo: `Analizando ${n} cámara${n === 1 ? "" : "s"}`,
    detalle: "Los modelos están mirando el video y van a avisar si detectan algo." };
}

function estadoMail(mail) {
  if (mail === null || mail === undefined) {
    return { nivel: "nose", corto: "Mail ?", titulo: "Mail sin dato",
      detalle: "Todavía no pude preguntarle al backend si el mail puede salir." };
  }
  if (mail.ok) {
    return { nivel: "ok", corto: "Mail", titulo: "Mail activo", detalle: "Las alertas también se mandan por mail a los contactos." };
  }
  return { nivel: "aviso", corto: "Sin mail", titulo: "Alertas sin mail",
    detalle: `Las alertas se guardan y se ven acá, pero NO le llega un mail a nadie. ${mail.motivo ?? ""}`.trim() };
}

export default function EstadoSistema({ conectado, servicio, mail }) {
  const items = [
    { clave: "panel", ...estadoPanel(conectado) },
    { clave: "ia", ...estadoModelos(servicio) },
    { clave: "mail", ...estadoMail(mail) },
  ];

  return (
    <ul className="estado-sistema" aria-label="Estado del sistema">
      {items.map((it) => (
        <li
          key={it.clave}
          className={`estado-sistema__item estado-sistema__item--${it.nivel}`}
          tabIndex={0}
          role="status"
          aria-live={it.nivel === "ok" ? "off" : "polite"}
          aria-label={`${it.titulo}. ${it.detalle}`}
        >
          <span className="estado-sistema__punto" aria-hidden="true" />
          <span className="estado-sistema__corto">{it.corto}</span>
          <span className="estado-sistema__tip" aria-hidden="true">
            <strong>{it.titulo}</strong>
            <span>{it.detalle}</span>
          </span>
        </li>
      ))}
    </ul>
  );
}
