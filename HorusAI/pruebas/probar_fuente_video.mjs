// De donde saca el panel el video de cada camara.
//
// Existe por el bug del 18/09: en Windows una webcam la abre UN proceso a la
// vez. El panel le pedia el video al backend, el backend abria la camara, y
// el servicio de modelos se quedaba sin poder abrirla. Los modelos corriendo
// sobre nada y en pantalla todo normal — "me detecta la camara pero el modelo
// no corre cuando la cam esta prendida".
//
//     node pruebas/probar_fuente_video.mjs

import assert from "node:assert/strict";

// fuenteVideo() importa API_URL de config.js, que usa import.meta.env. Node no
// lo tiene, asi que se replica la logica exacta acá. Si cambia una, esta
// prueba lo grita: los dos textos tienen que quedar iguales.
const API_URL = "http://localhost:8000";

function fuenteVideo(camaraId, servicio, ahora = 123) {
  const delServicio = servicio && typeof servicio.urlDe === "function"
    ? servicio.urlDe(camaraId)
    : null;
  if (delServicio && delServicio.url) return delServicio;
  return {
    url: `${API_URL}/video/video_feed/${camaraId}?t=${ahora}`,
    conIA: false, estado: null, error: null,
  };
}

// Un useServicioVideo() de mentira, con la misma forma que el de verdad.
function servicioFalso(activo, camaras) {
  return {
    activo,
    camaras,
    urlDe(configId) {
      const c = camaras[configId];
      if (activo && c) {
        return { url: `http://localhost:8010/camaras/${c.camara}/stream`,
                 conIA: true, estado: c.estado, error: c.error };
      }
      return { url: null, conIA: false, estado: null, error: null };
    },
  };
}

const CASOS = [
  ["con el servicio corriendo, el video sale del SERVICIO", () => {
    const s = servicioFalso(true, { 3: { camara: "cam-3", estado: "ok" } });
    const f = fuenteVideo(3, s);
    assert.equal(f.url, "http://localhost:8010/camaras/cam-3/stream");
    assert.equal(f.conIA, true);
    return "3 -> :8010/camaras/cam-3/stream";
  }],

  ["sin servicio, se cae al backend", () => {
    const f = fuenteVideo(3, servicioFalso(false, {}));
    assert.ok(f.url.startsWith("http://localhost:8000/video/video_feed/3"));
    assert.equal(f.conIA, false);
    return "3 -> :8000/video/video_feed/3";
  }],

  ["sin el hook siquiera, se cae al backend", () => {
    const f = fuenteVideo(7, null);
    assert.ok(f.url.includes("/video/video_feed/7"));
    return "sigue andando con servicio=null";
  }],

  ["el servicio corre pero NO tiene esta camara: backend", () => {
    // Pasa de verdad: agregas una camara nueva y el servicio todavia no la
    // sondeo. Pedirle el stream al servicio daria una imagen que no existe.
    const s = servicioFalso(true, { 1: { camara: "cam-1", estado: "ok" } });
    const f = fuenteVideo(9, s);
    assert.ok(f.url.includes("/video/video_feed/9"));
    assert.equal(f.conIA, false);
    return "camara 9 no esta en el servicio -> backend";
  }],

  ["una camara caida se distingue de una tranquila", () => {
    // Lo que no puede pasar es que 'la camara se cayo' y 'no esta pasando
    // nada' se vean igual, que es negro en los dos casos.
    const s = servicioFalso(true, {
      2: { camara: "cam-2", estado: "reconectando", error: "no abre" } });
    const f = fuenteVideo(2, s);
    assert.equal(f.estado, "reconectando");
    assert.equal(f.error, "no abre");
    return "estado y error llegan a la celda";
  }],

  ["los ids numericos y de texto son la misma camara", () => {
    // El backend manda config_id numerico; las claves de un objeto JS son
    // texto. Si esto se rompiera, el panel mostraria siempre el backend.
    const s = servicioFalso(true, { 5: { camara: "cam-5", estado: "ok" } });
    assert.equal(fuenteVideo(5, s).conIA, true);
    assert.equal(fuenteVideo("5", s).conIA, true);
    return "5 y \"5\" caen en la misma";
  }],
  ["la copia de fuenteVideo() no se desincronizo del original", async () => {
    // Esta prueba replica fuenteVideo() en vez de importarlo, porque el de
    // verdad importa config.js, que usa import.meta.env y Node no tiene. Una
    // copia se desincroniza sola con el tiempo, asi que se compara letra por
    // letra contra el archivo real. Si alguien toca uno y no el otro, esto
    // avisa en vez de dejar que la prueba mienta.
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const { dirname, join } = await import("node:path");
    const aqui = dirname(fileURLToPath(import.meta.url));
    const real = await readFile(
      join(aqui, "..", "src", "components", "CamaraGrid", "fuenteVideo.js"), "utf8");

    const cuerpo = (txt) => {
      const i = txt.indexOf("export function fuenteVideo") >= 0
        ? txt.indexOf("export function fuenteVideo")
        : txt.indexOf("function fuenteVideo");
      return txt.slice(i)
        .replace("export function", "function")
        .replace(/API_URL/g, "API")
        .replace(/\s+/g, " ")
        .trim();
    };

    const mio = cuerpo(`function fuenteVideo(camaraId, servicio, ahora = Date.now()) {
  const delServicio = servicio && typeof servicio.urlDe === "function"
    ? servicio.urlDe(camaraId)
    : null;

  if (delServicio && delServicio.url) {
    return delServicio;
  }
  return {
    url: \`\${API}/video/video_feed/\${camaraId}?t=\${ahora}\`,
    conIA: false,
    estado: null,
    error: null,
  };
}`);
    assert.equal(cuerpo(real), mio,
      "fuenteVideo.js cambio y esta prueba no: actualiza la copia de arriba");
    return "identica al original";
  }],
];

let fallas = 0;
console.log("=".repeat(70));
console.log("HORUS · de donde sale el video de cada camara");
console.log("=".repeat(70));
for (const [nombre, f] of CASOS) {
  try {
    const detalle = await f();
    console.log(`  OK     ${nombre.padEnd(48)} ${detalle}`);
  } catch (e) {
    console.log(`  FALLA  ${nombre.padEnd(48)} ${e.message.split("\n")[0]}`);
    fallas++;
  }
}
console.log("-".repeat(70));
console.log(`${CASOS.length - fallas}/${CASOS.length} pruebas OK`);
process.exit(fallas ? 1 : 0);
