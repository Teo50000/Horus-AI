/**
 * Filas "fantasma" con brillo mientras llega una lista del backend.
 * Tienen la forma de lo que va a aparecer, así el panel no salta al cargar.
 */
export default function SkeletonLista({ filas = 4, alto = 38, texto = "Cargando…" }) {
  return (
    <div className="skeleton-lista" role="status" aria-label={texto}>
      {Array.from({ length: filas }, (_, i) => (
        <div
          key={i}
          className="skeleton skeleton--fila"
          style={{ height: alto, opacity: 1 - i * (0.6 / filas) }}
        />
      ))}
    </div>
  );
}
