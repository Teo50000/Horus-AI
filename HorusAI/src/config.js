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

// Que canales de aviso estan vivos del lado del backend: websocket y mail.
// 22/09: el mail estaba apagado (no habia .env con las credenciales) y no lo
// decia nadie. El panel mostraba la alerta, la fila se guardaba, y el mail que
// tenia que despertar a alguien no salia nunca.
export const API_ESTADO  = `${API_URL}/estado`;

// El websocket por el que salen las alertas hacia el panel.
//
// Ojo con esto: el Dashboard se conectaba a /camaras/ws, que NO es el de las
// alertas. Funcionaba igual porque ConnectionManager.broadcast() reparte a
// todas las conexiones sin mirar en que bucket estan; o sea, andaba de
// carambola. Si algun dia broadcast pasa a filtrar por camara —que es lo
// razonable cuando haya varias— el panel se queda mudo sin un solo error.
export const WS_ALERTAS = `${WS_URL}/alertas/ws?camara_config_id=0`;

// El servicio de modelos (horus/00_servicio/servicio.py).
//
// 18/09: en Windows una webcam la abre UN proceso a la vez. El panel pedia el
// video al backend, el backend abria la camara, y entonces el servicio no
// podia abrirla: los modelos quedaban sin nada que mirar. El sintoma es "me
// detecta la camara pero el modelo no corre cuando la cam esta prendida".
//
// La solucion no es turnarse: el servicio YA sirve el video con las cajas
// dibujadas en su propio puerto. Asi que la camara la abre el servicio y
// nadie mas, y el panel muestra ese stream — que ademas es mejor, porque
// tiene las detecciones encima. Si el servicio no esta, se cae al video crudo
// del backend.
export const SERVICIO_URL = env.VITE_SERVICIO_URL ?? "http://localhost:8010";
