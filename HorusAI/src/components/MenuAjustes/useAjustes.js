import { useState } from "react";

/**
 * useAjustes
 * Lógica del panel de ajustes:
 *  - Lista de números de emergencia (agregar, editar, guardar)
 *  - Toggle de alerta en pantalla
 *  - Toggles de optimización de IA
 */

const NUMEROS_INICIALES = [
  { id: 1, nombre: "Numero1", telefono: "+5491125654003" },
  { id: 2, nombre: "Numero2", telefono: "+5491125654003" },
];

const CONFIG_IA_INICIAL = {
  incendios: true,
  desmayos: true,
  agresiones: true,
};

export function useAjustes() {
  // ── Números de emergencia ─────────────────────────────────────
  const [numeros, setNumeros] = useState(NUMEROS_INICIALES);
  // Cuál fila está en modo edición: null o el id del número
  const [editandoId, setEditandoId] = useState(null);

  const agregarNumero = () => {
    const nuevoId = Date.now();
    setNumeros((prev) => [
      ...prev,
      { id: nuevoId, nombre: `Numero${prev.length + 1}`, telefono: "" },
    ]);
    // Arranca en modo edición automáticamente
    setEditandoId(nuevoId);
  };

  const toggleEdicion = (id) => {
    setEditandoId((prev) => (prev === id ? null : id));
  };

  const actualizarNumero = (id, campo, valor) => {
    setNumeros((prev) =>
      prev.map((n) => (n.id === id ? { ...n, [campo]: valor } : n))
    );
  };

  const guardarNumero = (id) => {
    setEditandoId(null);
    // TODO: conectar con backend
    console.log("Guardar número:", numeros.find((n) => n.id === id));
  };

  // ── Toggle alerta en pantalla ─────────────────────────────────
  const [alertaEnPantalla, setAlertaEnPantalla] = useState(true);

  const toggleAlerta = () => {
    setAlertaEnPantalla((prev) => {
      const nuevo = !prev;
      // TODO: conectar con backend
      console.log("Alerta en pantalla:", nuevo);
      return nuevo;
    });
  };

  // ── Optimización IA ───────────────────────────────────────────
  const [configIA, setConfigIA] = useState(CONFIG_IA_INICIAL);

  const toggleIA = (clave) => {
    setConfigIA((prev) => {
      const nuevo = { ...prev, [clave]: !prev[clave] };
      // TODO: conectar con backend
      console.log("Config IA:", nuevo);
      return nuevo;
    });
  };

  return {
    // Números
    numeros,
    editandoId,
    agregarNumero,
    toggleEdicion,
    actualizarNumero,
    guardarNumero,
    // Alerta
    alertaEnPantalla,
    toggleAlerta,
    // IA
    configIA,
    toggleIA,
  };
}
