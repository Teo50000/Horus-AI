import { useState } from "react";
import "./MailConfig.css";

/**
 * MailConfig
 *
 * La cuenta DESDE la que salen los avisos.
 *
 * Por que esto tiene que estar en la aplicacion y no en un archivo: las
 * funciones de mail ya estaban escritas, la tabla de contactos tambien, y la
 * pantalla para cargarlos igual. Y aun asi no salia un solo mail, porque
 * faltaba la unica pieza que ningun codigo puede generar solo: la credencial
 * de la cuenta que manda. Esa pieza vivia en un .env que habia que escribir a
 * mano, afuera de la app. Para el que usa el sistema eso es identico a que
 * este roto.
 *
 * La clave se manda una sola vez al backend (127.0.0.1), se prueba contra el
 * servidor y se guarda. Nunca vuelve: lo unico que el backend contesta es si
 * se puede mandar y desde que direccion.
 */
export default function MailConfig({ mail, onGuardar, guardando, resultado }) {
  const [remitente, setRemitente] = useState("");
  const [clave, setClave] = useState("");
  const [abierto, setAbierto] = useState(false);

  const enviar = async (e) => {
    e.preventDefault();
    const r = await onGuardar(remitente.trim(), clave);
    if (r?.ok) {
      setClave("");          // no queda en memoria mas de lo necesario
      setAbierto(false);
    }
  };

  // Tres estados, no dos. `null` es "todavia no pude preguntarle al backend",
  // que NO es lo mismo que "anda": si esto mostrara verde cuando no hay
  // respuesta, un backend caido se veria como un sistema que avisa.
  const estado = mail === null ? "nose" : mail.ok ? "ok" : "no";

  return (
    <section className="ajustes-section">
      <h2 className="ajustes-section__title">Aviso por mail</h2>

      <div className={`mailcfg__estado mailcfg__estado--${estado}`}>
        <span className="mailcfg__punto" aria-hidden="true" />
        <span>
          {estado === "nose" && "No pude preguntarle al backend"}
          {estado === "no" && "APAGADO: no le llega un mail a nadie"}
          {estado === "ok" && `Activo, desde ${mail.remitente}`}
        </span>
      </div>

      {mail && !mail.ok && (
        <p className="mailcfg__motivo">{mail.motivo}</p>
      )}

      {/* Dos formas distintas de no avisarle a nadie, y se arreglan en
          lugares distintos. Con la cuenta configurada y cero contactos, el
          sistema puede mandar y no tiene a quien. */}
      {mail?.ok && !mail.hay_destinos && (
        <p className="mailcfg__motivo">
          Puede mandar, pero no hay ningun contacto con direccion cargada.
          Agregalos aca arriba.
        </p>
      )}

      {!abierto && (
        <button
          type="button"
          className="mailcfg__abrir"
          onClick={() => setAbierto(true)}
        >
          {mail?.ok ? "Cambiar la cuenta" : "Configurar la cuenta"}
        </button>
      )}

      {abierto && (
        <form className="mailcfg__form" onSubmit={enviar}>
          <label className="mailcfg__label">
            Cuenta desde la que se manda
            <input
              className="mailcfg__input"
              type="email"
              value={remitente}
              onChange={(e) => setRemitente(e.target.value)}
              placeholder="horus@gmail.com"
              autoComplete="off"
              required
            />
          </label>

          <label className="mailcfg__label">
            Contrasena de aplicacion
            <input
              className="mailcfg__input"
              type="password"
              value={clave}
              onChange={(e) => setClave(e.target.value)}
              placeholder="16 letras"
              autoComplete="off"
              required
            />
          </label>

          <p className="mailcfg__ayuda">
            No es la clave de tu Gmail: es una de 16 letras que genera Google
            aparte y solo sirve para esto. Se saca en{" "}
            <span className="mailcfg__url">
              myaccount.google.com/apppasswords
            </span>{" "}
            con la verificacion en dos pasos activada. Podes pegarla con los
            espacios, se los saco yo.
          </p>

          <div className="mailcfg__acciones">
            <button
              type="button"
              className="mailcfg__cancelar"
              onClick={() => { setAbierto(false); setClave(""); }}
              disabled={guardando}
            >
              Cancelar
            </button>
            <button
              type="submit"
              className="mailcfg__guardar"
              disabled={guardando || !remitente || !clave}
            >
              {guardando ? "Probando contra el servidor..." : "Probar y guardar"}
            </button>
          </div>

          {/* Se prueba el login ANTES de guardar. Una clave que el servidor
              rechaza no deja archivo escrito: si lo dejara, el cartel de
              arriba se apagaria y el sistema quedaria igual de mudo, pero ya
              sin nadie mirandolo. */}
          {resultado && (
            <p className={`mailcfg__resultado mailcfg__resultado--${
              resultado.ok ? "ok" : "no"}`}>
              {resultado.ok
                ? "Listo. El servidor acepto la clave y ya quedo guardada."
                : resultado.motivo}
            </p>
          )}
        </form>
      )}
    </section>
  );
}
