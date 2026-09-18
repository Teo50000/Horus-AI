import "./SearchBar.css";
import SearchIcon from "../../../assets/SearchIcon.svg";

export default function SearchBar({ value, onChange, onClear }) {
  return (
    <div className="search-bar">
      <input
        type="text"
        className="search-bar__input"
        placeholder="Buscar"
        value={value}
        onChange={onChange}
        aria-label="Buscar en historial"
      />
      {value && (
        <button
          className="search-bar__clear"
          onClick={onClear}
          aria-label="Limpiar búsqueda"
        >
          ✕
        </button>
      )}
      <img src={SearchIcon} alt="Buscar" className="search-bar__icon " />
    </div>
  );
}
