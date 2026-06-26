import "./EventoItem.css";

// { camara: "Camara 3", tipo: "Desmayo", fecha: "12/06/25" }
export default function EventoItem({ camara, tipo, fecha }) {
  return (
    <li className="evento-item">
      <span className="evento-item__camara">{camara}:</span>
      <span className="evento-item__tipo">{tipo}</span>
      <span className="evento-item__fecha">{fecha}</span>
    </li>
  );
}
