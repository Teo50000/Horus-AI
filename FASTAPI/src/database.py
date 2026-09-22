import os
import shutil
from datetime import datetime

from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import MetaData, inspect, text
from sqlalchemy.schema import CreateTable

engine = create_engine("sqlite:///horus.db")


def crear_tablas(motor=None):
    motor = motor or engine
    SQLModel.metadata.create_all(motor)
    migrar(motor)


def migrar(motor=None) -> list:
    """Agrega a las tablas viejas las columnas que el modelo ganó después.

    18/09, encontrado en el log del backend de Teo:

        sqlite3.OperationalError: no such column: alerta.version

    `SQLModel.metadata.create_all()` crea las TABLAS que faltan y nada más:
    si una tabla ya existe, no le toca una columna aunque el modelo haya
    cambiado. Su `horus.db` era de antes del contrato de alertas, así que la
    tabla `alerta` tenía 21 columnas de un esquema viejo y el modelo pide 32.
    Resultado: cada lectura de /alertas explotaba con un 500, el historial del
    panel quedaba vacío, y el error solo aparecía en una ventana que nadie
    mira.

    Borrar la base y empezar de cero también "arreglaba" el problema, y es
    exactamente lo que no hay que hacer con la base de un sistema de alertas:
    ahí está el historial de lo que pasó. Se agregan las columnas y listo.

    SQLite hace `ALTER TABLE ... ADD COLUMN` sin reescribir la tabla, así que
    esto es instantáneo por más filas que haya. Las columnas se agregan
    NULLABLE a propósito: las filas viejas no tienen ese dato y forzarlas a
    tener uno sería inventarlo.

    22/09 · lo que faltaba. Arriba decía que las columnas viejas "ocupan nada"
    y se podían dejar. Es falso para una: una columna sobrante declarada
    `NOT NULL` sin default no ocupa nada, PERO rechaza todos los INSERT que
    vengan. En la base de Teo quedaron tres de un esquema anterior
    —`ts_recibido`, `aportes_json`, `payload_json`— y el resultado era:

        sqlite3.IntegrityError: NOT NULL constraint failed: alerta.ts_recibido

    Cada alerta que mandaba Horus moría con 500. El emisor reintentaba, la
    spooleaba, y el panel se veía exactamente igual que en una noche
    tranquila. La última fila guardada era del 27/08. Un mes mudo.

    SQLite no sabe sacarle el NOT NULL a una columna, así que esa tabla hay
    que reconstruirla. No se pierde nada: las columnas viejas se vuelven a
    crear NULLABLES y se copian con sus datos (en `payload_json` está el
    mensaje original completo, que es historial de verdad). Antes de tocar
    nada se guarda una copia del archivo.

    Devuelve la lista de lo que agregó, para poder decirlo en voz alta.
    """
    motor = motor or engine
    agregadas = []
    inspector = inspect(motor)
    existentes = set(inspector.get_table_names())

    with motor.begin() as con:
        for tabla in SQLModel.metadata.sorted_tables:
            if tabla.name not in existentes:
                continue                      # la acaba de crear create_all
            hay = {c["name"] for c in inspector.get_columns(tabla.name)}
            for col in tabla.columns:
                if col.name in hay:
                    continue
                tipo = col.type.compile(motor.dialect)
                con.execute(text(
                    f'ALTER TABLE "{tabla.name}" ADD COLUMN "{col.name}" {tipo}'))
                agregadas.append(f"{tabla.name}.{col.name}")

    if agregadas:
        print(f"[base] {len(agregadas)} columna(s) agregadas a tablas que venían "
              f"de una versión anterior:")
        for a in agregadas:
            print(f"[base]   + {a}")

    # Y ahora lo que ADD COLUMN no puede arreglar.
    inspector = inspect(motor)
    bloqueadas = {}
    for tabla in SQLModel.metadata.sorted_tables:
        if tabla.name not in existentes:
            continue
        sobrantes = _sobrantes_que_bloquean(inspector, tabla)
        if sobrantes:
            bloqueadas[tabla.name] = [c["name"] for c in sobrantes]

    if bloqueadas:
        copia = _respaldo(motor)
        print("[base] " + "=" * 62)
        print("[base] Hay columnas de un esquema viejo declaradas NOT NULL que")
        print("[base] rechazan TODA alerta nueva. Reconstruyo la tabla sin")
        print("[base] perder nada (las columnas viejas quedan, nullables).")
        if copia:
            print(f"[base] Copia de seguridad: {os.path.basename(copia)}")
        for nombre, cols in bloqueadas.items():
            info = reconstruir(nombre, motor)
            print(f"[base]   {nombre}: {info['filas']} fila(s) intactas · "
                  f"destrabadas {', '.join(cols)}")
            agregadas.append(f"{nombre} (reconstruida)")
        print("[base] " + "=" * 62)

    return agregadas


# Columnas que cambiaron de nombre entre esquemas. Al reconstruir, el dato
# viejo se mete en la columna nueva en vez de quedar huérfano.
RENOMBRADAS = {"alerta": {"ts_recibido": "recibido_en"}}


def _ruta_base(motor=None) -> str:
    return str((motor or engine).url.database or "")


def _respaldo(motor=None) -> str:
    """Una copia del archivo antes de reconstruir. Es la base de un sistema de
    alertas: si algo sale mal, el historial tiene que seguir existiendo."""
    origen = _ruta_base(motor)
    if not origen or not os.path.exists(origen):
        return ""
    destino = "%s.respaldo-%s" % (origen, datetime.now().strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(origen, destino)
    return destino


def _sobrantes_que_bloquean(inspector, tabla) -> list:
    """Columnas que están en la base, NO están en el modelo, y son NOT NULL
    sin default: las únicas que impiden escribir."""
    delmodelo = {c.name for c in tabla.columns}
    fuera = []
    for c in inspector.get_columns(tabla.name):
        if c["name"] in delmodelo:
            continue
        if not c.get("nullable", True) and c.get("default") is None:
            fuera.append(c)
    return fuera


def _default_de(col):
    """El default que el modelo le daría a una fila nueva sin ese dato.

    Si no hay default declarado se usa el cero del tipo. No es inventar: la
    alternativa es tirar las filas viejas, y en la base de un sistema de
    alertas eso es borrar lo que pasó.
    """
    d = getattr(col, "default", None)
    if d is not None and getattr(d, "is_scalar", False):
        return d.arg
    try:
        py = col.type.python_type
    except NotImplementedError:
        return ""
    if py is bool:
        return False
    if py in (int, float):
        return py(0)
    return ""


def reconstruir(nombre_tabla: str, motor=None) -> dict:
    """Rehace una tabla para sacarle los NOT NULL que dejó un esquema viejo.

    Conserva TODAS las columnas, también las que el modelo ya no usa: las
    vuelve a crear nullables. Verifica que la cantidad de filas sea la misma
    antes de reemplazar; si no lo es, no reemplaza nada.
    """
    motor = motor or engine
    tabla = SQLModel.metadata.tables[nombre_tabla]
    inspector = inspect(motor)
    viejas = inspector.get_columns(nombre_tabla)
    nombres_viejos = [c["name"] for c in viejas]
    delmodelo = {c.name for c in tabla.columns}
    extras = [c for c in viejas if c["name"] not in delmodelo]

    tmp = "%s__nueva" % nombre_tabla
    md = MetaData()
    nueva = tabla.to_metadata(md, name=tmp)

    with motor.begin() as con:
        antes = con.execute(text('SELECT COUNT(*) FROM "%s"' % nombre_tabla)).scalar()
        con.execute(text('DROP TABLE IF EXISTS "%s"' % tmp))
        con.execute(CreateTable(nueva))
        for c in extras:                      # las viejas vuelven, nullables
            con.execute(text('ALTER TABLE "%s" ADD COLUMN "%s" %s'
                             % (tmp, c["name"],
                                c["type"].compile(motor.dialect))))

        # La tabla nueva tiene las columnas del modelo MÁS las viejas que
        # acabamos de re-agregar, así que todas las de la vieja existen allá.
        #
        # Las filas viejas llegaron acá por un ADD COLUMN, así que tienen NULL
        # en las columnas que el modelo sumó después y declara obligatorias
        # (`version`, `evidencia`, `recibido_en`...). Copiarlas tal cual choca
        # contra ese NOT NULL. Se rellenan con el default del PROPIO modelo,
        # que es el valor que habría tenido una fila nueva sin ese dato, y
        # antes se prueba el nombre viejo de la columna si lo hubo.
        renombres = {n: v for v, n in RENOMBRADAS.get(nombre_tabla, {}).items()
                     if v in nombres_viejos}
        columnas = {c.name: c for c in tabla.columns}
        selects, valores = [], {}
        for n in nombres_viejos:
            col = columnas.get(n)
            cadena = ['"%s"' % n]
            if n in renombres:
                cadena.append('"%s"' % renombres[n])
            if col is not None and not col.nullable and not col.primary_key:
                relleno = _default_de(col)
                if relleno is not None:
                    clave = "relleno_%s" % n
                    cadena.append(":%s" % clave)
                    valores[clave] = relleno
            selects.append(cadena[0] if len(cadena) == 1
                           else "COALESCE(%s)" % ", ".join(cadena))
        cols = ", ".join('"%s"' % n for n in nombres_viejos)
        con.execute(text('INSERT INTO "%s" (%s) SELECT %s FROM "%s"'
                         % (tmp, cols, ", ".join(selects), nombre_tabla)),
                    valores)

        despues = con.execute(text('SELECT COUNT(*) FROM "%s"' % tmp)).scalar()
        if despues != antes:
            con.execute(text('DROP TABLE IF EXISTS "%s"' % tmp))
            raise RuntimeError(
                "reconstruyendo %s: entraron %d filas y salieron %d. No toco "
                "nada." % (nombre_tabla, antes, despues))

        con.execute(text('DROP TABLE "%s"' % nombre_tabla))
        con.execute(text('ALTER TABLE "%s" RENAME TO "%s"' % (tmp, nombre_tabla)))

    return {"tabla": nombre_tabla, "filas": antes,
            "conservadas": [c["name"] for c in extras]}


def get_session():
    with Session(engine) as session:
        yield session
