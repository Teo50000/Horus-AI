import { useState, useEffect } from "react";

const API = "http://localhost:8000/camaras";

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

  // Carga inicial desde la DB
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
      console.log("Teléfono guardado:", numero);
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
      setEditandoId(data.id); // arranca en modo edición
    } catch (err) {
      console.error("Error al agregar teléfono:", err);
    }
  };

  // ── Toggle alerta en pantalla ─────────────────────────────────
  const [alertaEnPantalla, setAlertaEnPantalla] = useState(true);

  const toggleAlerta = () => {
    setAlertaEnPantalla((prev) => {
      const nuevo = !prev;
      console.log("Alerta en pantalla:", nuevo);
      // TODO: conectar con backend cuando haya endpoint
      return nuevo;
    });
  };

  // ── Optimización IA ───────────────────────────────────────────
  const [configIA, setConfigIA] = useState(CONFIG_IA_INICIAL);

  const toggleIA = (clave) => {
    setConfigIA((prev) => {
      const nuevo = { ...prev, [clave]: !prev[clave] };
      console.log("Config IA:", nuevo);
      // TODO: conectar con backend cuando haya endpoint
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
    alertaEnPantalla,
    toggleAlerta,
    configIA,
    toggleIA,
  };
}
