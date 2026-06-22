import "./Toggle.css";

export default function Toggle({ checked, onChange, label }) {
  return (
    <label className="toggle" aria-label={label}>
      <input
        type="checkbox"
        className="toggle__input"
        checked={checked}
        onChange={onChange}
      />
      <span className="toggle__track" />
    </label>
  );
}
