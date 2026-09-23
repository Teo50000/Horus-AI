import "./Toggle.css";

// `disabled` existe porque hay toggles que todavia no estan conectados a
// nada. Dejarlos clickeables seria dejar que alguien apague algo y crea que
// se apago. Ver el comentario en AjustesPanel.
export default function Toggle({ checked, onChange, label, disabled = false }) {
  return (
    <label className="toggle" aria-label={label}>
      <input
        type="checkbox"
        className="toggle__input"
        checked={checked}
        onChange={onChange}
        disabled={disabled}
      />
      <span className="toggle__track" />
    </label>
  );
}
