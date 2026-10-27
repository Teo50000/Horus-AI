//! Horus.exe — la ventana de escritorio de Horus.
//!
//! Antes: doble clic en HORUS.bat → tres ventanas negras (backend, modelos,
//! panel) + una pestaña de Edge con el panel. Ahora: doble clic en Horus.exe
//! → una sola ventana. Mientras arranca muestra solo el logo; cuando el panel
//! contesta, la misma ventana pasa a ser el panel. Lo que va diciendo
//! HORUS.bat no se muestra: queda en logs\arranque.txt. Si algo falla, la
//! ventana dice el motivo y ofrece abrir la carpeta de logs.
//!
//! La lógica de arranque NO está duplicada acá: la sigue teniendo HORUS.bat.
//! Esta app lo corre escondido con HORUS_APP=1, que le dice "sin ventanas
//! nuevas, sin pausas y sin abrir el navegador". Así, cuando se agregue una
//! cabeza o un chequeo nuevo, se toca un solo lugar y vale para los dos
//! caminos (el .bat a mano y este .exe).
//!
//! El panel se carga desde el servidor de Vite (localhost:1420), igual que en
//! el navegador. Por eso los cambios al panel se ven sin recompilar este .exe.
//!
//! Al cerrar la ventana pregunta, porque cerrar = dejar de vigilar: todo lo
//! que arrancó HORUS.bat vive en un "job" de Windows y se apaga con la app.
//! Si Horus ya estaba corriendo de antes (por ejemplo con HORUS.bat), la app
//! solo lo muestra y al cerrarla queda corriendo.
//!
//! Dos modos, según dónde esté el .exe:
//!
//!   - Adentro del proyecto (al lado de HORUS.bat, o más abajo, como en
//!     src-tauri\target\release): el modo de arriba. Todo, modelos incluidos.
//!   - Instalado suelto (el instalador de `tauri build`, sin el proyecto al
//!     lado): lo de antes. Arranca el backend empaquetado (backend.exe, el
//!     sidecar de PyInstaller) y muestra el panel que viaja adentro del .exe.
//!     Sin modelos: ese backend no los trae.

use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Manager, WindowEvent};

const PANEL_URL: &str = "http://localhost:1420";
/// Carpeta de datos de la app en %LOCALAPPDATA% (es el `identifier`).
#[cfg(all(windows, target_env = "gnu"))]
const CARPETA_APP: &str = "com.tauri-app.horusai";
const PUERTO_PANEL: u16 = 1420;
const PUERTO_BACKEND: u16 = 8000;
const PUERTO_MODELOS: u16 = 8010;

/// Los puertos de las tres partes. Al cerrar se apaga lo que escuche en
/// ellos (si lo arrancamos nosotros), además del job: ver `apagar_todo`.
const PUERTOS_HORUS: [u16; 3] = [PUERTO_BACKEND, PUERTO_MODELOS, PUERTO_PANEL];

/// Cuántas líneas del arranque se guardan para la pantalla. Después de que
/// aparece el panel nadie las mira, pero el lector sigue vaciando el caño
/// para que ningún proceso se trabe escribiendo en él.
const MAX_LINEAS: usize = 2000;

/// Cuánto se sigue esperando al panel en total. El primer arranque en una PC
/// nueva instala dependencias (pip, npm) y puede tardar varios minutos.
const ESPERA_MAXIMA: Duration = Duration::from_secs(20 * 60);

#[derive(Clone, Serialize, PartialEq)]
#[serde(tag = "tipo", content = "detalle", rename_all = "snake_case")]
enum Fase {
    Arrancando,
    Listo,
    Error(String),
}

struct Lineas {
    total: usize,
    buf: VecDeque<String>,
}

struct Estado {
    raiz: Mutex<Option<PathBuf>>,
    lineas: Mutex<Lineas>,
    archivo: Mutex<Option<File>>,
    fase: Mutex<Fase>,
    backend: AtomicBool,
    panel: AtomicBool,
    /// Arrancamos procesos nosotros: al cerrar se apagan (y se pregunta).
    propio: AtomicBool,
    /// Hay un cartel de "¿apagar?" abierto.
    preguntando: AtomicBool,
    /// Ya dijo que sí: la próxima vez que se pida cerrar, se cierra.
    saliendo: AtomicBool,
    /// Handle del job de Windows (0 = no hay).
    job: AtomicUsize,
    /// PID de HORUS.bat, por si el job no se pudo armar.
    pid: AtomicU32,
    /// Puertos que ya estaban ocupados antes de arrancar: son de otro (por
    /// ejemplo un HORUS.bat a mano) y al cerrar NO se tocan.
    ajenos: Mutex<Vec<u16>>,
    /// (puerto, pid) de lo que arrancamos, a medida que aparece escuchando.
    /// Se guarda en disco para limpiar huérfanos si Horus.exe se cae.
    registrados: Mutex<Vec<(u16, u32)>>,
}

impl Estado {
    fn nuevo() -> Self {
        Estado {
            raiz: Mutex::new(None),
            lineas: Mutex::new(Lineas {
                total: 0,
                buf: VecDeque::new(),
            }),
            archivo: Mutex::new(None),
            fase: Mutex::new(Fase::Arrancando),
            backend: AtomicBool::new(false),
            panel: AtomicBool::new(false),
            propio: AtomicBool::new(false),
            preguntando: AtomicBool::new(false),
            saliendo: AtomicBool::new(false),
            job: AtomicUsize::new(0),
            pid: AtomicU32::new(0),
            ajenos: Mutex::new(Vec::new()),
            registrados: Mutex::new(Vec::new()),
        }
    }

    fn linea(&self, texto: impl Into<String>) {
        let texto = texto.into();
        if let Ok(mut f) = self.archivo.lock() {
            if let Some(f) = f.as_mut() {
                let _ = writeln!(f, "{texto}");
                let _ = f.flush();
            }
        }
        let mut l = self.lineas.lock().unwrap();
        l.total += 1;
        l.buf.push_back(texto);
        while l.buf.len() > MAX_LINEAS {
            l.buf.pop_front();
        }
    }

    fn fallar(&self, motivo: impl Into<String>) {
        let motivo = motivo.into();
        let mut f = self.fase.lock().unwrap();
        if *f != Fase::Listo {
            *f = Fase::Error(motivo);
        }
    }
}

// ---------------------------------------------------------------------------
//  Comandos que usa la pantalla de arranque (splash/index.html)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct Respuesta {
    total: usize,
    lineas: Vec<String>,
    fase: Fase,
    backend: bool,
    panel: bool,
    raiz: Option<String>,
}

/// Lo que pasó desde la línea `desde`. La pantalla pregunta cada ~250 ms: es
/// más simple y más robusto que eventos (no hay carrera entre "me suscribo" y
/// "ya se emitieron las primeras líneas").
#[tauri::command]
fn estado(desde: usize, est: tauri::State<'_, Arc<Estado>>) -> Respuesta {
    let (total, lineas) = {
        let l = est.lineas.lock().unwrap();
        let primera = l.total - l.buf.len();
        let desde = desde.max(primera).min(l.total);
        let lineas = l.buf.iter().skip(desde - primera).cloned().collect();
        (l.total, lineas)
    };
    Respuesta {
        total,
        lineas,
        fase: est.fase.lock().unwrap().clone(),
        backend: est.backend.load(Ordering::Relaxed),
        panel: est.panel.load(Ordering::Relaxed),
        raiz: est
            .raiz
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.display().to_string()),
    }
}

#[tauri::command]
fn abrir_logs(est: tauri::State<'_, Arc<Estado>>) {
    let raiz = est.raiz.lock().unwrap().clone();
    if let Some(raiz) = raiz {
        let logs = raiz.join("logs");
        let destino = if logs.is_dir() { logs } else { raiz };
        let _ = Command::new("explorer.exe").arg(destino).spawn();
    }
}

#[tauri::command]
fn cerrar_app(app: AppHandle, est: tauri::State<'_, Arc<Estado>>) {
    est.saliendo.store(true, Ordering::SeqCst);
    apagar_todo(&est);
    app.exit(0);
}

// ---------------------------------------------------------------------------
//  Arranque
// ---------------------------------------------------------------------------

fn es_raiz(p: &Path) -> bool {
    p.join("HORUS.bat").is_file() && p.join("FASTAPI").is_dir()
}

/// La carpeta del proyecto: la del .exe o alguna de más arriba (así anda
/// igual en la raíz del repo que en HorusAI\src-tauri\target\release).
/// HORUS_RAIZ la pisa, por si alguna vez hace falta.
fn buscar_raiz() -> Option<PathBuf> {
    if let Ok(r) = std::env::var("HORUS_RAIZ") {
        let r = PathBuf::from(r);
        if es_raiz(&r) {
            return Some(r);
        }
    }
    let exe = std::env::current_exe().ok()?;
    let exe = fs::canonicalize(&exe).unwrap_or(exe);
    exe.ancestors()
        .skip(1)
        .take(8)
        .find(|p| es_raiz(p))
        .map(limpiar_ruta)
}

/// `canonicalize` en Windows devuelve `\\?\C:\...`; cmd.exe no lo acepta
/// como carpeta de trabajo.
fn limpiar_ruta(p: &Path) -> PathBuf {
    let s = p.display().to_string();
    match s.strip_prefix(r"\\?\") {
        Some(resto) if !resto.starts_with("UNC") => PathBuf::from(resto),
        _ => p.to_path_buf(),
    }
}

/// ¿Alguien escucha en ese puerto? Se prueba IPv4 e IPv6: Vite en Windows
/// suele escuchar solo en ::1 ("localhost"), el backend en 127.0.0.1.
fn puerto_abierto(puerto: u16) -> bool {
    let mut destinos: Vec<SocketAddr> = vec![
        SocketAddr::from(([127, 0, 0, 1], puerto)),
        SocketAddr::from(([0, 0, 0, 0, 0, 0, 0, 1], puerto)),
    ];
    if let Ok(it) = ("localhost", puerto).to_socket_addrs() {
        for a in it {
            if !destinos.contains(&a) {
                destinos.push(a);
            }
        }
    }
    destinos
        .iter()
        .any(|a| TcpStream::connect_timeout(a, Duration::from_millis(300)).is_ok())
}

fn arrancar(app: AppHandle, est: Arc<Estado>) {
    let Some(raiz) = buscar_raiz() else {
        modo_empaquetado(app, est);
        return;
    };
    *est.raiz.lock().unwrap() = Some(raiz.clone());

    // Lo que dice el arranque queda también en logs\arranque.txt: después de
    // que aparece el panel, esta pantalla ya no se ve.
    let logs = raiz.join("logs");
    let _ = fs::create_dir_all(&logs);
    if let Ok(f) = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(logs.join("arranque.txt"))
    {
        *est.archivo.lock().unwrap() = Some(f);
    }

    // Lo que haya quedado prendido de un Horus.exe anterior que se cayó (o
    // que se cerró antes de este arreglo): se reconoce porque el mismo PID
    // sigue escuchando en el mismo puerto que quedó anotado.
    limpiar_huerfanos(&est);

    // ¿Ya está andando? (por ejemplo, lo arrancaste con HORUS.bat). Entonces
    // no se arranca nada: se muestra ese, y al cerrar la ventana sigue vivo.
    if puerto_abierto(PUERTO_PANEL) && puerto_abierto(PUERTO_BACKEND) {
        est.backend.store(true, Ordering::Relaxed);
        est.panel.store(true, Ordering::Relaxed);
        est.linea("Horus ya estaba corriendo: muestro ese.");
        est.linea("Como no lo arranqué yo, al cerrar esta ventana sigue corriendo.");
        mostrar_panel(&app, &est, Panel::Vite);
        return;
    }

    est.linea(format!("Carpeta: {}", raiz.display()));
    anotar_ajenos(&est);
    let mut hijo = match lanzar_bat(&raiz) {
        Ok(h) => h,
        Err(e) => {
            est.linea(format!("No pude ejecutar HORUS.bat: {e}"));
            est.fallar(format!("No pude ejecutar HORUS.bat: {e}"));
            return;
        }
    };
    est.propio.store(true, Ordering::SeqCst);
    est.pid.store(hijo.id(), Ordering::SeqCst);
    match job::atar(&hijo) {
        Ok(h) => est.job.store(h, Ordering::SeqCst),
        Err(e) => est.linea(format!(
            "(aviso: {e}. Al cerrar intento apagar todo igual, por PID.)"
        )),
    }

    if let Some(salida) = hijo.stdout.take() {
        let est2 = est.clone();
        thread::spawn(move || leer(salida, est2));
    }
    {
        let est2 = est.clone();
        thread::spawn(move || registrar_procesos(est2));
    }

    let inicio = Instant::now();
    let mut termino: Option<(Instant, Option<i32>)> = None;
    loop {
        if est.saliendo.load(Ordering::SeqCst) {
            return;
        }
        est.backend
            .store(puerto_abierto(PUERTO_BACKEND), Ordering::Relaxed);
        if puerto_abierto(PUERTO_PANEL) {
            est.panel.store(true, Ordering::Relaxed);
            mostrar_panel(&app, &est, Panel::Vite);
            return;
        }

        if termino.is_none() {
            if let Ok(Some(st)) = hijo.try_wait() {
                termino = Some((Instant::now(), st.code()));
            }
        }
        // HORUS.bat terminó y el panel no aparece. Unos segundos de gracia
        // (Vite puede estar terminando de levantar) y se dice por qué. Se
        // sigue mirando igual: si el panel aparece tarde, se muestra.
        if let Some((cuando, codigo)) = termino {
            if cuando.elapsed() > Duration::from_secs(8) {
                let motivo = match codigo {
                    Some(0) | None => "El panel no arrancó. El motivo está en \
                                       logs\\arranque.txt y logs\\panel.txt."
                        .to_string(),
                    Some(c) => format!(
                        "HORUS.bat se cortó antes de levantar el panel (código {c}). \
                         El motivo está al final de logs\\arranque.txt."
                    ),
                };
                est.fallar(motivo);
            }
        }
        if inicio.elapsed() > ESPERA_MAXIMA {
            est.fallar("Pasaron 20 minutos y el panel no aparece. Mirá logs\\panel.txt.");
            return;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

enum Panel {
    /// El de Vite (localhost:1420): el código del panel tal como está hoy.
    Vite,
    /// El que viaja adentro del .exe (dist/, armado en `tauri build`).
    Empaquetado,
}

fn mostrar_panel(app: &AppHandle, est: &Estado, cual: Panel) {
    *est.fase.lock().unwrap() = Fase::Listo;
    est.linea("Panel listo.");
    if let Some(w) = app.get_webview_window("main") {
        let url = match cual {
            Panel::Vite => tauri::Url::parse(PANEL_URL).ok(),
            // splash/index.html → index.html, en el mismo origen que sirve Tauri.
            Panel::Empaquetado => w.url().ok().and_then(|u| u.join("/index.html").ok()),
        };
        if let Some(url) = url {
            let _ = w.navigate(url);
        }
    }
}

/// Sin el proyecto al lado: lo que hacía la app antes. Arranca el backend
/// empaquetado (backend.exe, que `tauri build` deja al lado del .exe) y
/// muestra el panel que viaja adentro.
fn modo_empaquetado(app: AppHandle, est: Arc<Estado>) {
    let backend = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.join("backend.exe")))
        .filter(|p| p.is_file());

    let Some(backend) = backend else {
        est.linea("No encuentro la carpeta de Horus.");
        est.fallar(
            "No encuentro la carpeta de Horus. Horus.exe tiene que estar en la carpeta \
             del proyecto, al lado de HORUS.bat. Para tenerlo en el escritorio hacé un \
             acceso directo (clic derecho → Enviar a → Escritorio) en vez de copiarlo.",
        );
        return;
    };

    est.linea("Sin la carpeta del proyecto al lado: modo instalado.");
    est.linea("Arranco el backend empaquetado. Los modelos NO vienen en este modo.");

    limpiar_huerfanos(&est);
    anotar_ajenos(&est);
    if puerto_abierto(PUERTO_BACKEND) {
        est.linea("Ya hay un backend escuchando en el 8000: uso ese.");
    } else {
        {
            let est2 = est.clone();
            thread::spawn(move || registrar_procesos(est2));
        }
        match lanzar_oculto(&backend) {
            Ok(mut hijo) => {
                est.propio.store(true, Ordering::SeqCst);
                est.pid.store(hijo.id(), Ordering::SeqCst);
                match job::atar(&hijo) {
                    Ok(h) => est.job.store(h, Ordering::SeqCst),
                    Err(e) => est.linea(format!("(aviso: {e})")),
                }
                if let Some(salida) = hijo.stdout.take() {
                    let est2 = est.clone();
                    thread::spawn(move || leer(salida, est2));
                }
            }
            Err(e) => {
                est.fallar(format!("No pude arrancar {}: {e}", backend.display()));
                return;
            }
        }
    }

    let inicio = Instant::now();
    while inicio.elapsed() < Duration::from_secs(40) {
        if est.saliendo.load(Ordering::SeqCst) {
            return;
        }
        if puerto_abierto(PUERTO_BACKEND) {
            est.backend.store(true, Ordering::Relaxed);
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
    if !est.backend.load(Ordering::Relaxed) {
        est.linea("El backend no contestó en 40 s: abro el panel igual (va a decir desconectado).");
    }
    est.panel.store(true, Ordering::Relaxed);
    mostrar_panel(&app, &est, Panel::Empaquetado);
}

/// Lee la salida de HORUS.bat línea por línea. pip y npm dibujan barras de
/// progreso con `\r`: de cada línea se queda el último tramo, que es el que
/// se vería en la consola.
fn leer<R: Read>(r: R, est: Arc<Estado>) {
    let mut lector = BufReader::new(r);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match lector.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => break,
            Ok(_) => {
                let s = String::from_utf8_lossy(&buf);
                let s = s.trim_end_matches(['\r', '\n']);
                let s = s.rsplit('\r').next().unwrap_or("");
                est.linea(s.to_string());
            }
        }
    }
}

#[cfg(windows)]
fn lanzar_bat(raiz: &Path) -> std::io::Result<Child> {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;

    let cmd = std::env::var("ComSpec").unwrap_or_else(|_| "cmd.exe".into());
    let bat = raiz.join("HORUS.bat");
    // `/s /c ""ruta" 2>&1"`: con /s cmd saca SOLO el primer y el último par
    // de comillas, y queda `"ruta" 2>&1`. Así una ruta con espacios no se
    // rompe, y stderr viaja por el mismo caño que stdout.
    Command::new(cmd)
        .raw_arg(format!("/d /s /c \"\"{}\" 2>&1\"", bat.display()))
        .current_dir(raiz)
        .env("HORUS_APP", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
        .spawn()
}

#[cfg(windows)]
fn lanzar_oculto(exe: &Path) -> std::io::Result<Child> {
    use std::os::windows::process::CommandExt;
    let mut c = Command::new(exe);
    if let Some(d) = exe.parent() {
        c.current_dir(d);
    }
    c.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(0x0800_0000)
        .spawn()
}

#[cfg(not(windows))]
fn lanzar_oculto(exe: &Path) -> std::io::Result<Child> {
    Command::new(exe).stdout(Stdio::piped()).spawn()
}

#[cfg(not(windows))]
fn lanzar_bat(_raiz: &Path) -> std::io::Result<Child> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "HORUS.bat es de Windows",
    ))
}

/// Apaga todo lo que arrancó HORUS.bat (backend, modelos, panel, y lo que
/// hayan lanzado ellos). Lo que ya estaba corriendo antes no se toca.
///
/// El job solo no alcanza: el Python de la Microsoft Store (el de esta PC)
/// arranca por un "alias de ejecución" y Windows lo crea FUERA del job de
/// quien lo llamó. Así que además se apaga lo que escuche en los puertos de
/// Horus, salvo los que ya estaban ocupados antes de arrancar.
fn apagar_todo(est: &Estado) {
    if !est.propio.load(Ordering::SeqCst) {
        return;
    }
    let h = est.job.load(Ordering::SeqCst);
    if h != 0 {
        job::terminar(h);
    } else {
        let pid = est.pid.load(Ordering::SeqCst);
        if pid != 0 {
            matar_arbol(pid);
        }
    }

    let ajenos = est.ajenos.lock().unwrap().clone();
    let mut pids: Vec<u32> = puertos::duenos(&PUERTOS_HORUS)
        .into_iter()
        .filter(|(p, _)| !ajenos.contains(p))
        .map(|(_, pid)| pid)
        .collect();
    // Y lo anotado que siga vivo en su puerto (por si cambió de dueño, no).
    let vivos = puertos::duenos(&PUERTOS_HORUS);
    for r in est.registrados.lock().unwrap().iter() {
        if vivos.contains(r) {
            pids.push(r.1);
        }
    }
    pids.sort();
    pids.dedup();
    let propio = std::process::id();
    for pid in pids.into_iter().filter(|&p| p != 0 && p != propio) {
        matar_arbol(pid);
    }
    let _ = fs::remove_file(archivo_procesos());
}

/// Puertos ya ocupados antes de arrancar nada: no son nuestros.
fn anotar_ajenos(est: &Estado) {
    let ocupados: Vec<u16> = puertos::duenos(&PUERTOS_HORUS)
        .into_iter()
        .map(|(p, _)| p)
        .collect();
    *est.ajenos.lock().unwrap() = ocupados;
}

/// Dónde quedan anotados los procesos de esta corrida. Si Horus.exe se cae,
/// la próxima vez que abre los encuentra y los apaga.
fn archivo_procesos() -> PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("com.tauri-app.horusai").join("procesos.txt")
}

fn guardar_procesos(lista: &[(u16, u32)]) {
    let ruta = archivo_procesos();
    if let Some(d) = ruta.parent() {
        let _ = fs::create_dir_all(d);
    }
    let texto: String = lista.iter().map(|(p, pid)| format!("{p} {pid}\n")).collect();
    let _ = fs::write(ruta, texto);
}

/// Va anotando (puerto, pid) de lo que arrancamos a medida que se pone a
/// escuchar. Los modelos tardan: pueden aparecer un minuto después del panel.
fn registrar_procesos(est: Arc<Estado>) {
    let inicio = Instant::now();
    while inicio.elapsed() < Duration::from_secs(30 * 60) {
        if est.saliendo.load(Ordering::SeqCst) {
            return;
        }
        let ajenos = est.ajenos.lock().unwrap().clone();
        let mut cambio = false;
        {
            let mut reg = est.registrados.lock().unwrap();
            for (p, pid) in puertos::duenos(&PUERTOS_HORUS) {
                if !ajenos.contains(&p) && !reg.contains(&(p, pid)) {
                    reg.push((p, pid));
                    cambio = true;
                }
            }
            if cambio {
                guardar_procesos(&reg);
            }
        }
        let todos = {
            let reg = est.registrados.lock().unwrap();
            PUERTOS_HORUS
                .iter()
                .all(|p| ajenos.contains(p) || reg.iter().any(|(q, _)| q == p))
        };
        if todos {
            return;
        }
        thread::sleep(Duration::from_secs(2));
    }
}

/// Apaga lo que quedó de un Horus.exe anterior: solo si el MISMO pid sigue
/// escuchando en el MISMO puerto que quedó anotado. Un HORUS.bat a mano no
/// anota nada, así que nunca se toca.
fn limpiar_huerfanos(est: &Estado) {
    let Ok(texto) = fs::read_to_string(archivo_procesos()) else {
        return;
    };
    let anotados: Vec<(u16, u32)> = texto
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect();
    let vivos = puertos::duenos(&PUERTOS_HORUS);
    let mut apagados = 0;
    for r in anotados.iter().filter(|r| vivos.contains(r)) {
        matar_arbol(r.1);
        apagados += 1;
    }
    if apagados > 0 {
        est.linea(format!(
            "Apagué {apagados} proceso(s) que habían quedado prendidos de la vez anterior."
        ));
        thread::sleep(Duration::from_secs(2));
    }
    let _ = fs::remove_file(archivo_procesos());
}

#[cfg(windows)]
fn matar_arbol(pid: u32) {
    use std::os::windows::process::CommandExt;
    let _ = Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .creation_flags(0x0800_0000)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(windows))]
fn matar_arbol(_pid: u32) {}

// ---------------------------------------------------------------------------
//  Quién escucha en qué puerto (la tabla TCP de Windows)
// ---------------------------------------------------------------------------

#[cfg(windows)]
mod puertos {
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCP6ROW_OWNER_PID, MIB_TCPROW_OWNER_PID,
        TCP_TABLE_OWNER_PID_LISTENER,
    };

    const AF_INET: u32 = 2;
    const AF_INET6: u32 = 23;

    /// (puerto, pid) de los procesos que escuchan en alguno de `puertos`,
    /// en IPv4 y en IPv6 (Vite suele escuchar solo en ::1).
    pub fn duenos(puertos: &[u16]) -> Vec<(u16, u32)> {
        let mut out = Vec::new();
        for af in [AF_INET, AF_INET6] {
            let mut tam: u32 = 0;
            unsafe {
                GetExtendedTcpTable(
                    std::ptr::null_mut(),
                    &mut tam,
                    0,
                    af,
                    TCP_TABLE_OWNER_PID_LISTENER,
                    0,
                );
            }
            if tam == 0 {
                continue;
            }
            // Vec<u32> para que quede alineado; con margen por si la tabla
            // creció entre las dos llamadas.
            let mut buf: Vec<u32> = vec![0; (tam as usize + 4096) / 4];
            let mut tam = (buf.len() * 4) as u32;
            let r = unsafe {
                GetExtendedTcpTable(
                    buf.as_mut_ptr() as *mut core::ffi::c_void,
                    &mut tam,
                    0,
                    af,
                    TCP_TABLE_OWNER_PID_LISTENER,
                    0,
                )
            };
            if r != 0 {
                continue;
            }
            let n = buf[0] as usize;
            unsafe {
                let base = buf.as_ptr().add(1) as *const u8;
                for i in 0..n {
                    let (puerto, pid) = if af == AF_INET {
                        let f = std::ptr::read_unaligned(
                            (base as *const MIB_TCPROW_OWNER_PID).add(i),
                        );
                        (u16::from_be((f.dwLocalPort & 0xffff) as u16), f.dwOwningPid)
                    } else {
                        let f = std::ptr::read_unaligned(
                            (base as *const MIB_TCP6ROW_OWNER_PID).add(i),
                        );
                        (u16::from_be((f.dwLocalPort & 0xffff) as u16), f.dwOwningPid)
                    };
                    if puertos.contains(&puerto) {
                        out.push((puerto, pid));
                    }
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }
}

#[cfg(not(windows))]
mod puertos {
    pub fn duenos(_puertos: &[u16]) -> Vec<(u16, u32)> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
//  Job de Windows: todo lo que arranca HORUS.bat muere con Horus.exe
// ---------------------------------------------------------------------------
//
// Con KILL_ON_JOB_CLOSE, cuando se cierra el último handle del job (o sea,
// cuando Horus.exe termina, incluso si se cuelga y lo matás desde el
// administrador de tareas), Windows mata a todos los procesos del job. Los
// hijos y nietos de HORUS.bat entran solos: un proceso nuevo hereda el job de
// su padre. Sin esto, cerrar la app dejaba python y node vivos y escondidos,
// sin ventana donde apretar Ctrl+C.
//
// La app en sí NO entra al job: WebView2 arma sus propios jobs para el
// sandbox y no vale la pena el riesgo de anidarlos.
//
// OJO: el job no atrapa al Python de la Microsoft Store (ver apagar_todo).

#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub fn atar(hijo: &Child) -> Result<usize, String> {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return Err(format!("no pude crear el job ({})", GetLastError()));
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                return Err(format!("no pude configurar el job ({})", GetLastError()));
            }
            let proceso = hijo.as_raw_handle() as windows_sys::Win32::Foundation::HANDLE;
            if AssignProcessToJobObject(job, proceso) == 0 {
                return Err(format!("no pude atar HORUS.bat al job ({})", GetLastError()));
            }
            // El handle NO se cierra: tiene que vivir lo que vive Horus.exe.
            Ok(job as usize)
        }
    }

    pub fn terminar(h: usize) {
        unsafe {
            TerminateJobObject(h as windows_sys::Win32::Foundation::HANDLE, 0);
        }
    }
}

#[cfg(not(windows))]
mod job {
    pub fn atar(_hijo: &std::process::Child) -> Result<usize, String> {
        Err("sin jobs fuera de Windows".into())
    }
    pub fn terminar(_h: usize) {}
}

// ---------------------------------------------------------------------------
//  "¿Apagar Horus?"
// ---------------------------------------------------------------------------

#[cfg(windows)]
fn confirmar_apagar(dueno: isize) -> bool {
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        MessageBoxW, IDYES, MB_DEFBUTTON2, MB_ICONWARNING, MB_SETFOREGROUND, MB_YESNO,
    };
    fn ancho(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }
    let texto = ancho(
        "Si cerrás Horus se apagan los modelos, el backend y el panel:\n\
         deja de mirar las cámaras y no sale ninguna alerta.\n\n\
         ¿Apagar Horus?",
    );
    let titulo = ancho("Horus");
    let r = unsafe {
        MessageBoxW(
            dueno as _,
            texto.as_ptr(),
            titulo.as_ptr(),
            MB_YESNO | MB_ICONWARNING | MB_DEFBUTTON2 | MB_SETFOREGROUND,
        )
    };
    r == IDYES
}

#[cfg(not(windows))]
fn confirmar_apagar(_dueno: isize) -> bool {
    true
}

fn hwnd_de(w: &tauri::Window) -> isize {
    #[cfg(windows)]
    {
        w.hwnd().map(|h| h.0 as isize).unwrap_or(0)
    }
    #[cfg(not(windows))]
    {
        let _ = w;
        0
    }
}

// ---------------------------------------------------------------------------
//  WebView2Loader.dll adentro del .exe (solo compilación GNU)
// ---------------------------------------------------------------------------
//
// Compilado con Rust para Windows "de verdad" (MSVC, `npx tauri build`), el
// cargador de WebView2 va estático y esto no existe. Compilado desde Linux
// (toolchain GNU), webview2-com-sys pide WebView2Loader.dll al lado del .exe.
// Para no dejar una DLL suelta en la carpeta del proyecto (que alguien borra,
// y el .exe deja de abrir con un error de Windows que no dice nada), la DLL
// viaja adentro del .exe, se escribe en %LOCALAPPDATA%\com.tauri-app.horusai y se
// carga antes de que Tauri la necesite. El link es con carga diferida
// (dlltool -y), así que Windows no la busca al arrancar el proceso, y cuando
// se llama por primera vez encuentra la que ya está cargada.

#[cfg(all(windows, target_env = "gnu"))]
mod cargador_webview2 {
    use std::os::windows::ffi::OsStrExt;
    use std::path::PathBuf;
    use windows_sys::Win32::Foundation::GetLastError;
    use windows_sys::Win32::System::LibraryLoader::LoadLibraryW;

    static DLL: &[u8] = include_bytes!(env!("HORUS_WEBVIEW2_DLL"));

    pub fn preparar() -> Result<(), String> {
        let dir_exe = std::env::current_exe()
            .ok()
            .and_then(|e| e.parent().map(|p| p.to_path_buf()));
        let mut carpetas: Vec<PathBuf> = Vec::new();
        // Si alguien la dejó al lado del .exe, esa manda (y no se pisa).
        if let Some(d) = &dir_exe {
            if d.join("WebView2Loader.dll").is_file() {
                carpetas.push(d.clone());
            }
        }
        if let Some(d) = std::env::var_os("LOCALAPPDATA") {
            carpetas.push(PathBuf::from(d).join(super::CARPETA_APP));
        }
        carpetas.push(std::env::temp_dir().join("horus-webview2"));

        let mut ultimo = String::from("sin carpeta donde escribir");
        for dir in carpetas {
            let ruta = dir.join("WebView2Loader.dll");
            let es_la_del_exe = dir_exe.as_ref() == Some(&dir);
            let al_dia = std::fs::read(&ruta).map(|b| b == DLL).unwrap_or(false);
            if !es_la_del_exe && !al_dia {
                if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&ruta, DLL)) {
                    ultimo = format!("{}: {e}", ruta.display());
                    continue;
                }
            }
            let ancho: Vec<u16> = ruta.as_os_str().encode_wide().chain(Some(0)).collect();
            let h = unsafe { LoadLibraryW(ancho.as_ptr()) };
            if !h.is_null() {
                return Ok(());
            }
            ultimo = format!("{}: no carga (error {})", ruta.display(), unsafe { GetLastError() });
        }
        Err(ultimo)
    }
}

#[cfg(windows)]
fn cartel_error(texto: &str) {
    use windows_sys::Win32::UI::WindowsAndMessaging::{MessageBoxW, MB_ICONERROR, MB_OK};
    let t: Vec<u16> = texto.encode_utf16().chain(Some(0)).collect();
    let c: Vec<u16> = "Horus".encode_utf16().chain(Some(0)).collect();
    unsafe {
        MessageBoxW(std::ptr::null_mut(), t.as_ptr(), c.as_ptr(), MB_OK | MB_ICONERROR);
    }
}

// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    #[cfg(all(windows, target_env = "gnu"))]
    if let Err(e) = cargador_webview2::preparar() {
        cartel_error(&format!(
            "Horus no pudo preparar WebView2Loader.dll, que necesita para abrir la ventana.\n\n{e}\n\n\
             Mientras tanto podés arrancar con HORUS.bat."
        ));
        return;
    }

    let est0 = Arc::new(Estado::nuevo());

    tauri::Builder::default()
        // Un solo Horus a la vez: el segundo doble clic trae al frente la
        // ventana que ya está, en vez de arrancar todo otra vez y chocar con
        // los puertos.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_opener::init())
        .manage(est0.clone())
        .setup(move |app| {
            let handle = app.handle().clone();
            let est = est0.clone();
            thread::spawn(move || arrancar(handle, est));
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                let est = window.state::<Arc<Estado>>().inner().clone();
                if est.saliendo.load(Ordering::SeqCst) || !est.propio.load(Ordering::SeqCst) {
                    return;
                }
                // Si falló el arranque no hay nada que vigilar: se cierra
                // directo (y se apaga lo que haya quedado a medias).
                let fallo = matches!(*est.fase.lock().unwrap(), Fase::Error(_));
                if fallo {
                    est.saliendo.store(true, Ordering::SeqCst);
                    apagar_todo(&est);
                    return;
                }
                api.prevent_close();
                if est.preguntando.swap(true, Ordering::SeqCst) {
                    return;
                }
                let app = window.app_handle().clone();
                let dueno = hwnd_de(window);
                // El cartel va en otro hilo: es modal y no puede frenar el
                // bucle de eventos de la ventana.
                thread::spawn(move || {
                    let si = confirmar_apagar(dueno);
                    est.preguntando.store(false, Ordering::SeqCst);
                    if si {
                        est.saliendo.store(true, Ordering::SeqCst);
                        apagar_todo(&est);
                        app.exit(0);
                    }
                });
            }
        })
        .invoke_handler(tauri::generate_handler![estado, abrir_logs, cerrar_app])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
