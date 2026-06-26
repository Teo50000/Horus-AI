import NumeroItem from "./NumeroItem/NumeroItem";
import AddButton from "./AddButton/AddButton";
import Toggle from "./Toggle/Toggle";
import CloseButton from "../CloseButton/CloseButton";
import "./AjustesPanel.css";

export default function AjustesPanel({
  onClose,
  numeros,
  editandoId,
  agregarNumero,
  toggleEdicion,
  actualizarNumero,
  guardarNumero,
  alertaEnPantalla,
  toggleAlerta,
  configIA,
  toggleIA,
}) {
  return (
    <div className="ajustes-panel">

      <CloseButton onClick={onClose} />

      <div className="ajustes-panel__content">

        <section className="ajustes-section">
          <h2 className="ajustes-section__title">Notificación de eventos</h2>

          <div className="ajustes-numeros-lista">
            {numeros.map((n) => (
              <NumeroItem
                key={n.id}
                numero={n}
                editando={editandoId === n.id}
                onToggleEdicion={toggleEdicion}
                onActualizar={actualizarNumero}
                onGuardar={guardarNumero}
              />
            ))}
          </div>

          <AddButton onClick={agregarNumero} label="Agregar número" />

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
