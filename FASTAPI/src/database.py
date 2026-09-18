from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy import inspect, text

engine = create_engine("sqlite:///horus.db")


def crear_tablas():
    SQLModel.metadata.create_all(engine)
    migrar()


def migrar() -> list:
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

    Las columnas viejas que el modelo ya no usa se dejan donde están. Ocupan
    nada, y tirarlas significaría reescribir la tabla para perder datos.

    Devuelve la lista de lo que agregó, para poder decirlo en voz alta.
    """
    agregadas = []
    inspector = inspect(engine)
    existentes = set(inspector.get_table_names())

    with engine.begin() as con:
        for tabla in SQLModel.metadata.sorted_tables:
            if tabla.name not in existentes:
                continue                      # la acaba de crear create_all
            hay = {c["name"] for c in inspector.get_columns(tabla.name)}
            for col in tabla.columns:
                if col.name in hay:
                    continue
                tipo = col.type.compile(engine.dialect)
                con.execute(text(
                    f'ALTER TABLE "{tabla.name}" ADD COLUMN "{col.name}" {tipo}'))
                agregadas.append(f"{tabla.name}.{col.name}")

    if agregadas:
        print(f"[base] {len(agregadas)} columna(s) agregadas a tablas que venían "
              f"de una versión anterior:")
        for a in agregadas:
            print(f"[base]   + {a}")
    return agregadas


def get_session():
    with Session(engine) as session:
        yield session
