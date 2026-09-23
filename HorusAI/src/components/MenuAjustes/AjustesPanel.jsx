import ContactoItem from "./ContactoItem/ContactoItem";
import MailConfig from "./MailConfig/MailConfig";
import AddButton from "./AddButton/AddButton";
import RemoveButton from "../RemoveButton/RemoveButton";
import Toggle from "./Toggle/Toggle";
import CloseButton from "../CloseButton/CloseButton";
import "./AjustesPanel.css";

export default function AjustesPanel({
  onClose,
  contactos, cargando,
  editandoId, agregarContacto, toggleEdicion, actualizarContacto, guardarContacto,
  modoBorrado, seleccionadosIds, toggleModoBorrado, toggleSeleccion,
  confirmarBorrado, cancelarBorrado,
  alertaEnPantalla, toggleAlerta,
  configIA, toggleIA, IA_CONECTADA,
  mail, guardarMail, guardandoMail, resultadoMail,
}) {
  return (
    <div className="ajustes-panel">
      <CloseButton onClick={onClose} />
      <div className="ajustes-panel__content">

        <section className="ajustes-section">
          <h2 className="ajustes-section__title">Notificación de eventos</h2>

          {cargando ? (
            <p className="ajustes-panel__cargando">Cargando teléfonos...</p>
          ) : (
            <div className="ajustes-contactos-lista">
              {contactos.map((n) => (
                <div key={n.id} className="ajustes-contacto-row">
                  {/* Checkbox visible solo en modo borrado */}
                  {modoBorrado && (
                    <input
                      type="checkbox"
                      className="ajustes-checkbox"
                      checked={seleccionadosIds.has(n.id)}
                      onChange={() => toggleSeleccion(n.id)}
                    />
                  )}
                  <ContactoItem
                    contacto={n}
                    editando={!modoBorrado && editandoId === n.id}
                    onToggleEdicion={toggleEdicion}
                    onActualizar={actualizarContacto}
                    onGuardar={guardarContacto}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Botones + y − */}
          <div className="ajustes-acciones">
            {modoBorrado ? (
              <>
                <button className="ajustes-acciones__cancelar" onClick={cancelarBorrado}>
                  Cancelar
                </button>
                <button
                  className="ajustes-acciones__confirmar"
                  onClick={confirmarBorrado}
                  disabled={seleccionadosIds.size === 0}
                >
                  Aceptar
                </button>
              </>
            ) : (
              <>
                <AddButton onClick={agregarContacto} label="Agregar contacto" />
                <RemoveButton onClick={toggleModoBorrado} label="Eliminar contactos" />
              </>
            )}
          </div>

          <div className="ajustes-toggle-row">
            <span className="ajustes-toggle-label">Alerta de eventos en pantalla</span>
            <Toggle checked={alertaEnPantalla} onChange={toggleAlerta} label="Alerta en pantalla" />
          </div>
        </section>

        <MailConfig
          mail={mail}
          onGuardar={guardarMail}
          guardando={guardandoMail}
          resultado={resultadoMail}
        />

        <section className="ajustes-section">
          <h2 className="ajustes-section__title">Optimización de la IA</h2>
          {/* Estos tres NO apagan ningun modelo todavia: el servicio decide
              que cabezas carga por los flags con los que arranca. Apagar
              "incendios" aca dejaba el sistema detectando incendios igual, y
              al que lo apagaba lo dejaba creyendo que no. Un control que
              miente es peor que un control que no esta, asi que quedan a la
              vista pero deshabilitados y con el motivo escrito. */}
          {["incendios", "desmayos", "agresiones"].map((k) => (
            <div
              key={k}
              className={`ajustes-toggle-row ${
                IA_CONECTADA ? "" : "ajustes-toggle-row--muerto"}`}
            >
              <span className="ajustes-toggle-label">
                Detección de {k === "incendios" ? "incendios"
                  : k === "desmayos" ? "desmayos" : "agresiones"}
              </span>
              <Toggle
                checked={configIA[k]}
                onChange={() => toggleIA(k)}
                label={k}
                disabled={!IA_CONECTADA}
              />
            </div>
          ))}
          {!IA_CONECTADA && (
            <p className="ajustes-nota-honesta">
              Estos tres todavia no apagan nada: que modelos corren se decide
              al arrancar HORUS.bat. Los dejamos a la vista para no esconder
              que faltan, pero no queremos que apagues algo y siga prendido.
            </p>
          )}
        </section>

      </div>
    </div>
  );
}
