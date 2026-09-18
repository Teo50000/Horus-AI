// Un solo lugar donde vive la direccion del backend.
//
// 18/09: este archivo ya existia con las dos constantes de abajo, pero no lo
// importaba NADIE: los ocho componentes tenian "http://localhost:8000" escrito
// a mano. Cambiar el puerto significaba editar ocho archivos y acordarse de
// todos. Ahora todos importan de aca.
//
// Las variables de entorno son para cuando el backend no corre en esta misma
// maquina (otra PC de la red, un servidor). Se definen en un archivo .env al
// lado de package.json, con el prefijo VITE_ que exige Vite:
//
//     VITE_API_URL=http://192.168.0.50:8000
//     VITE_WS_URL=ws://192.168.0.50:8000
//
// Sin .env, los valores de abajo son los de siempre y no cambia nada.

const env = import.meta.env ?? {};

export const API_URL = env.VITE_API_URL ?? "http://localhost:8000";
export const WS_URL  = env.VITE_WS_URL  ?? "ws://localhost:8000";

// Los routers del backend (main.py: include_router con prefijo).
export const API_CAMARAS = `${API_URL}/camaras`;
export const API_VIDEO   = `${API_URL}/video`;
export const API_ALERTAS = `${API_URL}/alertas`;

// El websocket por el que salen las alertas hacia el panel.
//
// Ojo con esto: el Dashboard se conectaba a /camaras/ws, que NO es el de las
// alertas. Funcionaba igual porque ConnectionManager.broadcast() reparte a
// todas las conexiones sin mirar en que bucket estan; o sea, andaba de
// carambola. Si algun dia broadcast pasa a filtrar por camara —que es lo
// razonable cuando haya varias— el panel se queda mudo sin un solo error.
export const WS_ALERTAS = `${WS_URL}/alertas/ws?camara_config_id=0`;
