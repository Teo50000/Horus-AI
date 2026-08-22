import NumeroItem from "./NumeroItem/NumeroItem";
import AddButton from "./AddButton/AddButton";
import RemoveButton from "../RemoveButton/RemoveButton";
import Toggle from "./Toggle/Toggle";
import CloseButton from "../CloseButton/CloseButton";
import "./AjustesPanel.css";

export default function AjustesPanel({
  onClose,
  numeros, cargando,
  editandoId, agregarNumero, toggleEdicion, actualizarNumero, guardarNumero,
  modoBorrado, seleccionadosIds, toggleModoBorrado, toggleSeleccion,
  confirmarBorrado, cancelarBorrado,
  alertaEnPantalla, toggleAlerta,
  configIA, toggleIA,
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
            <div className="ajustes-numeros-lista">
              {numeros.map((n) => (
                <div key={n.id} className="ajustes-numero-row">
                  {/* Checkbox visible solo en modo borrado */}
                  {modoBorrado && (
                    <input
                      type="checkbox"
                      className="ajustes-checkbox"
                      checked={seleccionadosIds.has(n.id)}
                      onChange={() => toggleSeleccion(n.id)}
                    />
                  )}
                  <NumeroItem
                    numero={n}
                    editando={!modoBorrado && editandoId === n.id}
                    onToggleEdicion={toggleEdicion}
                    onActualizar={actualizarNumero}
                    onGuardar={guardarNumero}
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
                <AddButton onClick={agregarNumero} label="Agregar número" />
                <RemoveButton onClick={toggleModoBorrado} label="Eliminar números" />
              </>
            )}
          </div>

          <div className="ajustes-toggle-row">
            <span className="ajustes-toggle-label">Alerta de eventos en pantalla</span>
            <Toggle checked={alertaEnPantalla} onChange={toggleAlerta} label="Alerta en pantalla" />
          </div>
        </section>

        <section className="ajustes-section">
          <h2 className="ajustes-section__title">Optimización de la IA</h2>
          <div className="ajustes-toggle-row">
            <span className="ajustes-toggle-label">Detección de incendios</span>
            <Toggle checked={configIA.incendios} onChange={() => toggleIA("incendios")} label="Incendios" />
          </div>
          <div className="ajustes-toggle-row">
            <span className="ajustes-toggle-label">Detección de desmayos</span>
            <Toggle checked={configIA.desmayos} onChange={() => toggleIA("desmayos")} label="Desmayos" />
          </div>
          <div className="ajustes-toggle-row">
            <span className="ajustes-toggle-label">Detección de agresiones</span>
            <Toggle checked={configIA.agresiones} onChange={() => toggleIA("agresiones")} label="Agresiones" />
          </div>
        </section>

      </div>
    </div>
  );
}
