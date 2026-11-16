import CeldaCamara from "./CeldaCamara/CeldaCamara";
import logo from "../../assets/Logo.svg";
import "./CamaraGrid.css";

// El slot central (índice 4 en una grilla 3x3) es el logo
// Layout visual: [0][1][2] / [3][LOGO][4] / [5][6][7]
// Mapeamos los 8 slots a las 9 posiciones del grid saltando el centro

const POSICIONES_GRID = [0, 1, 2, 3, 5, 6, 7, 8]; // posición 4 = logo

export default function CamaraGrid({ slots, onNavegar, onVaciar, onAgregar, servicio }) {
  return (
    // La zona ocupa todo lo que deja el menú; la grilla adentro mantiene
    // 16:9 y se centra (ver CamaraGrid.css). Así, al abrir o cerrar el menú
    // las celdas se achican o agrandan SIN cambiar de forma ni recortar el
    // video.
    <main className="camara-grid-zona" aria-label="Cámaras">
      <div className="camara-grid">
        {POSICIONES_GRID.map((posGrid, slotIdx) => (
          <div
            key={posGrid}
            className="camara-grid__celda"
            style={{ gridArea: `pos${posGrid}` }}
          >
            <CeldaCamara
              slot={slots[slotIdx]}
              slotIdx={slotIdx}
              onNavegar={onNavegar}
              onVaciar={onVaciar}
              onAgregar={onAgregar}
              servicio={servicio}
            />
          </div>
        ))}

        {/* Logo central */}
        <div className="camara-grid__celda camara-grid__logo" style={{ gridArea: "pos4" }}>
          <img src={logo} alt="" className="camara-grid__logo-img" />
        </div>
      </div>
    </main>
  );
}
