import { useState, useEffect } from "react";

import { API_CAMARAS, API_CONFIG } from "../../config";
const API = API_CAMARAS;

const DELETE_TELEFONO = (id) => `${API}/emergencia/${id}`;

const CONFIG_IA_INICIAL = {
  incendios: true,
  desmayos: true,
  agresiones: true,
};

export function useAjustes() {
  // ── Números de emergencia ─────────────────────────────────────
  const [contactos, setContactos]       = useState([]);
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
      .then((data) => setContactos(data))
      .catch((err) => console.error("Error al cargar teléfonos:", err))
      .finally(() => setCargando(false));
  }, []);

  const toggleEdicion = (id) =>
    setEditandoId((prev) => (prev === id ? null : id));

  const actualizarContacto = (id, campo, valor) =>
    setContactos((prev) =>
      prev.map((n) => (n.id === id ? { ...n, [campo]: valor } : n))
    );

  const guardarContacto = async (id) => {
    setEditandoId(null);
    const contacto = contactos.find((n) => n.id === id);
    if (!contacto) return;
    try {
      await fetch(`${API}/emergencia/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ telefono: contacto.telefono, nombre: contacto.nombre }),
      });
    } catch (err) {
      console.error("Error al guardar teléfono:", err);
    }
  };

  const agregarContacto = async () => {
    const nuevo = { nombre: `Contacto${contactos.length + 1}`, telefono: "" };
    try {
      const res = await fetch(`${API}/emergencia`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(nuevo),
      });
      const data = await res.json();
      setContactos((prev) => [...prev, data]);
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
      setContactos((prev) => prev.filter((n) => !ids.includes(n.id)));
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

  const toggleAlerta = () => setAlertaEnPantalla((prev) => !prev);

  // ── Aviso por mail ────────────────────────────────────────────
  //
  // Esto faltaba, y era lo unico que faltaba. Las funciones de mail ya
  // estaban escritas y la pantalla para cargar contactos tambien; lo que no
  // habia era donde poner la cuenta DESDE la que se manda. Esa credencial
  // vivia en un archivo .env que habia que escribir a mano, afuera de la
  // aplicacion. Para el que usa el sistema eso es indistinguible de que este
  // roto: los contactos cargados, la pantalla completa, y no llega nada.
  //
  // La clave viaja al backend (127.0.0.1) una sola vez, se prueba contra el
  // servidor y se guarda. Nunca vuelve: el GET contesta si se puede mandar y
  // desde que direccion, nada mas.
  const [mail, setMail] = useState(null);      // {ok, remitente, motivo, hay_destinos}
  const [guardandoMail, setGuardandoMail] = useState(false);
  const [resultadoMail, setResultadoMail] = useState(null); // {ok, motivo}

  const leerEstadoMail = () => {
    fetch(`${API_CONFIG}/mail`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null))
      .then(setMail)
      .catch(() => setMail(null));   // null = no se, distinto de "anda"
  };

  useEffect(leerEstadoMail, []);

  const guardarMail = async (remitente, clave) => {
    setGuardandoMail(true);
    setResultadoMail(null);
    try {
      const r = await fetch(`${API_CONFIG}/mail`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ remitente, clave }),
      });
      const data = await r.json();
      setResultadoMail(data);
      if (data.ok) leerEstadoMail();
      return data;
    } catch (e) {
      const data = { ok: false, motivo: "no pude hablar con el backend" };
      setResultadoMail(data);
      return data;
    } finally {
      setGuardandoMail(false);
    }
  };

  // ── Optimización IA ───────────────────────────────────────────
  const [configIA, setConfigIA] = useState(CONFIG_IA_INICIAL);

  // OJO: esto NO apaga ningun modelo todavia.
  //
  // El toggle mueve un estado de React y hace un console.log. Nada de esto
  // llega al servicio de modelos, que decide que cabezas carga por los flags
  // con los que arranca (--segmentacion, --agresion, --caidas). O sea que
  // apagar "Deteccion de incendios" aca deja el sistema detectando incendios
  // exactamente igual, y el que lo apago se queda creyendo que no.
  //
  // Un control que miente es peor que un control que no esta: el que apaga
  // algo y ve que se apaga, se va tranquilo. Por eso la pantalla los muestra
  // deshabilitados y avisa, hasta que el servicio acepte prender y apagar
  // cabezas en caliente. Ver `servicio.py`, que hoy solo lee los flags al
  // arrancar.
  const IA_CONECTADA = false;

  const toggleIA = (clave) => {
    if (!IA_CONECTADA) return;
    setConfigIA((prev) => ({ ...prev, [clave]: !prev[clave] }));
  };

  return {
    contactos,
    cargando,
    editandoId,
    agregarContacto,
    toggleEdicion,
    actualizarContacto,
    guardarContacto,
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
    IA_CONECTADA,
    // mail
    mail,
    guardarMail,
    guardandoMail,
    resultadoMail,
  };
}
