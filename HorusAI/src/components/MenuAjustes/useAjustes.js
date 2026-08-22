import { useState, useEffect } from "react";

const API = "http://localhost:8000/camaras";

const DELETE_TELEFONO = (id) => `${API}/emergencia/${id}`;

const CONFIG_IA_INICIAL = {
  incendios: true,
  desmayos: true,
  agresiones: true,
};

export function useAjustes() {
  // ── Números de emergencia ─────────────────────────────────────
  const [numeros, setNumeros]       = useState([]);
  const [cargando, setCargando]     = useState(true);
  const [editandoId, setEditandoId] = useState(null);

  // ── Estado de modo borrado ────────────────────────────────────
  const [modoBorrado, setModoBorrado]           = useState(false);
  const [seleccionadosIds, setSeleccionadosIds] = useState(new Set());

  useEffect(() => {
    fetch(`${API}/emergencia`)
      .then((res) => {
        if (res.status === 404) return [];
        if (!res.ok) throw new Error(`Error ${res.status}`);
        return res.json();
      })
      .then((data) => setNumeros(data))
      .catch((err) => console.error("Error al cargar teléfonos:", err))
      .finally(() => setCargando(false));
  }, []);

  const toggleEdicion = (id) =>
    setEditandoId((prev) => (prev === id ? null : id));

  const actualizarNumero = (id, campo, valor) =>
    setNumeros((prev) =>
      prev.map((n) => (n.id === id ? { ...n, [campo]: valor } : n))
    );

  const guardarNumero = async (id) => {
    setEditandoId(null);
    const numero = numeros.find((n) => n.id === id);
    if (!numero) return;
    try {
      await fetch(`${API}/emergencia/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telefono: numero.telefono, nombre: numero.nombre }),
      });
    } catch (err) {
      console.error("Error al guardar teléfono:", err);
    }
  };

  const agregarNumero = async () => {
    const nuevo = { nombre: `Numero${numeros.length + 1}`, telefono: "" };
    try {
      const res = await fetch(`${API}/emergencia`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nuevo),
      });
      const data = await res.json();
      setNumeros((prev) => [...prev, data]);
      setEditandoId(data.id);
    } catch (err) {
      console.error("Error al agregar teléfono:", err);
    }
  };

  // ── Modo borrado ──────────────────────────────────────────────
  const toggleModoBorrado = () => {
    setModoBorrado((prev) => !prev);
    setSeleccionadosIds(new Set()); // limpia selección al entrar/salir
  };

  const toggleSeleccion = (id) => {
    setSeleccionadosIds((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const confirmarBorrado = async () => {
    const ids = [...seleccionadosIds];
    try {
      await Promise.all(
        ids.map((id) =>
          fetch(DELETE_TELEFONO(id), { method: "DELETE" })
        )
      );
      setNumeros((prev) => prev.filter((n) => !ids.includes(n.id)));
    } catch (err) {
      console.error("Error al borrar teléfonos:", err);
    }
    setModoBorrado(false);
    setSeleccionadosIds(new Set());
  };

  const cancelarBorrado = () => {
    setModoBorrado(false);
    setSeleccionadosIds(new Set());
  };

  // ── Toggle alerta en pantalla ─────────────────────────────────
  const [alertaEnPantalla, setAlertaEnPantalla] = useState(true);

  const toggleAlerta = () => {
    setAlertaEnPantalla((prev) => {
      const nuevo = !prev;
      console.log("Alerta en pantalla:", nuevo);
      return nuevo;
    });
  };

  // ── Optimización IA ───────────────────────────────────────────
  const [configIA, setConfigIA] = useState(CONFIG_IA_INICIAL);

  const toggleIA = (clave) => {
    setConfigIA((prev) => {
      const nuevo = { ...prev, [clave]: !prev[clave] };
      console.log("Config IA:", nuevo);
      return nuevo;
    });
  };

  return {
    numeros,
    cargando,
    editandoId,
    agregarNumero,
    toggleEdicion,
    actualizarNumero,
    guardarNumero,
    // borrado
    modoBorrado,
    seleccionadosIds,
    toggleModoBorrado,
    toggleSeleccion,
    confirmarBorrado,
    cancelarBorrado,
    // resto
    alertaEnPantalla,
    toggleAlerta,
    configIA,
    toggleIA,
  };
}
