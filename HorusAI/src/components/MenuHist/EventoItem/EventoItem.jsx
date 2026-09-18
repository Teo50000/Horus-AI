import "./EventoItem.css";

// { camara: "Deposito", tipo: "Paquete abandonado", fecha: "24/08/26 12:00" }
//
// 18/09: hasta hoy el historial estaba siempre vacio, asi que nadie vio una
// fila de verdad. Apenas empezo a cargar lo guardado se noto que las tres
// cosas en una linea no entran: el nombre de la camara quedaba en "cam-d..."
// y con "Paquete abandonado" desaparecia del todo. Van en dos lineas: arriba
// QUE paso, que es lo que uno busca al recorrer la lista; abajo donde y
// cuando.
export default function EventoItem({ camara, tipo, fecha }) {
  return (
    <li className="evento-item">
      <span className="evento-item__tipo">{tipo}</span>
      <span className="evento-item__fecha">{fecha}</span>
      <span className="evento-item__camara" title={camara}>{camara}</span>
    </li>
  );
}
