import { API_URL } from "../../config";

/**
 * De dónde sale el video de una cámara.
 *
 * 18/09: la celda pedía siempre `${API}/video/video_feed/{id}`, o sea que la
 * cámara la abría el BACKEND. En Windows una webcam la abre UN proceso a la
 * vez, así que el servicio de modelos no podía abrirla: los modelos corriendo
 * sobre nada, y en pantalla todo normal. El síntoma es "me detecta la cámara
 * pero el modelo no corre cuando la cam está prendida".
 *
 * Turnarse no sirve. La cámara la abre el servicio —que es el que la necesita
 * para analizar— y el video sale de ahí, ya con las cajas dibujadas. Si el
 * servicio no está corriendo, se cae al video crudo del backend.
 *
 * Está en su propio archivo, y no adentro del componente, para poder probarlo
 * sin levantar React.
 *
 * @param camaraId  el id de CamaraConfig en el backend
 * @param servicio  lo que devuelve useServicioVideo(), o null
 */
export function fuenteVideo(camaraId, servicio, ahora = Date.now()) {
  const delServicio = servicio && typeof servicio.urlDe === "function"
    ? servicio.urlDe(camaraId)
    : null;

  if (delServicio && delServicio.url) {
    return delServicio;
  }
  return {
    url: `${API_URL}/video/video_feed/${camaraId}?t=${ahora}`,
    conIA: false,
    estado: null,
    error: null,
  };
}
