#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_mail.py - contesta una sola pregunta: si pasa algo, le llega a alguien?

22/09. Hasta hoy la respuesta era que no, y lo peor era que no se notaba.
Medido en la base de Teo: 41 alertas de severidad 2 guardadas, 5 eventos
distintos, CERO mails. No existia el archivo .env con las credenciales, asi
que todos los envios morian en el login, la excepcion la comia un try y
quedaba un print en una ventana que nadie mira. El panel mostraba la alerta,
la fila se guardaba, la API contestaba 200. Un sistema que no aviso se veia
igual que uno que aviso.

Este script mira las tres cosas que tienen que estar, por separado, porque
fallan por separado:

    1. las credenciales (FASTAPI\\.env)
    2. los contactos con direccion de mail cargada
    3. que el 465 de gmail sea alcanzable desde esta maquina

    python bin\\probar_mail.py             solo revisa, no manda nada
    python bin\\probar_mail.py --mandar    manda uno de prueba de verdad

La contrasena no se imprime nunca, ni entera ni cortada. Si te falta, la
generas vos en https://myaccount.google.com/apppasswords y la escribis en el
.env: no hace falta que se la pases a nadie.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = os.path.join(RAIZ, "FASTAPI")
sys.path.insert(0, FAST)

# El engine del backend es "sqlite:///horus.db": relativo al directorio actual.
# Importandolo desde otro lado, SQLite no encuentra la base... y no falla: crea
# una vacia, sin contactos y sin historial, y este script informaba
# tranquilamente que no hay a quien avisarle. Una base equivocada que se
# comporta como una base vacia es la peor respuesta posible para la pregunta
# "le llega a alguien". Por eso el chdir va ANTES de importar nada de src.
if os.path.isdir(FAST):
    os.chdir(FAST)


def _titulo(t: str) -> None:
    print()
    print(t)
    print("-" * len(t))


def main() -> int:
    ap = argparse.ArgumentParser(description="Revisar el aviso por mail.")
    ap.add_argument("--mandar", action="store_true",
                    help="mandar un mail de prueba de verdad")
    ap.add_argument("--a", help="mandarlo a esta direccion en vez de a los "
                                "contactos cargados")
    args = ap.parse_args()

    print("HORUS - el aviso por mail")
    print("=" * 62)

    # ---------------------------------------------------------------- 1
    _titulo("1. Credenciales")
    try:
        from src.services import email_service as es
    except ImportError as e:
        print("  No pude importar el backend: %s" % e)
        print("  Corre esto desde la raiz del repo.")
        return 2

    listo, motivo = es.estado_mail()
    print("  archivo .env : %s" % ("esta" if os.path.exists(es.RUTA_ENV)
                                   else "NO EXISTE"))
    print("  ruta         : %s" % es.RUTA_ENV)
    print("  remitente    : %s" % (es.EMAIL_SENDER or "(sin definir)"))
    print("  contrasena   : %s" % ("definida" if es.EMAIL_PASSWORD
                                   else "SIN DEFINIR"))
    print("  servidor     : %s:%d" % (es.SMTP_HOST, es.SMTP_PORT))
    print()
    if listo:
        print("  OK - se puede mandar.")
    else:
        print("  APAGADO - %s" % motivo)
        print()
        print("  Se configura desde el panel: Ajustes -> Aviso por mail.")
        print("  (o con  python bin\\configurar_mail.py  si estas en consola)")
        print("  EMAIL_PASSWORD no es la clave de tu Gmail: es una")
        print("  'contrasena de aplicacion' de 16 letras que genera Google en")
        print("  https://myaccount.google.com/apppasswords")

    # ---------------------------------------------------------------- 2
    _titulo("2. A quien se le avisa")
    destinos = []
    try:
        from sqlmodel import Session, select
        from src.database import engine
        from src.models.camara_model import NumeroEmergencia
        print("  base: %s" % os.path.join(FAST, "horus.db"))
        with Session(engine) as s:
            contactos = s.exec(select(NumeroEmergencia)).all()
            for c in contactos:
                d = c.direccion_mail()
                print("  %-18s %s" % (
                    (c.nombre or "(sin nombre)")[:18],
                    d if d else "SIN DIRECCION DE MAIL (%r)" % (
                        c.telefono or c.email)))
                if d:
                    destinos.append(d)
    except Exception as e:
        print("  No pude leer los contactos: %s" % e)

    if not destinos:
        print()
        print("  NO HAY NADIE con direccion cargada. Aunque el mail este")
        print("  configurado, no le va a llegar a ninguna persona.")
        print("  Se agregan en el panel, en Ajustes.")

    # ---------------------------------------------------------------- 3
    _titulo("3. Se puede salir al servidor de mail?")
    try:
        with socket.create_connection((es.SMTP_HOST, es.SMTP_PORT), timeout=8):
            print("  OK - %s:%d responde desde esta maquina."
                  % (es.SMTP_HOST, es.SMTP_PORT))
            alcanzable = True
    except Exception as e:
        alcanzable = False
        print("  NO se puede: %s" % e)
        print("  Suele ser el firewall o la red del lugar bloqueando el 465.")

    # ---------------------------------------------------------------- 4
    _titulo("Resumen")
    faltan = []
    if not listo:
        faltan.append("las credenciales")
    if not destinos and not args.a:
        faltan.append("un contacto con direccion")
    if not alcanzable:
        faltan.append("salida al 465")
    if faltan:
        print("  Si pasa algo AHORA, no le llega a nadie: falta %s."
              % ", ".join(faltan))
    else:
        print("  Todo en orden. Proba con --mandar para confirmarlo de punta")
        print("  a punta.")

    if not args.mandar:
        print()
        print("(No mande ningun mail. Agregale --mandar para eso.)")
        return 0 if not faltan else 1

    # ---------------------------------------------------------------- 5
    _titulo("Mandando uno de prueba")
    if not listo:
        print("  No lo intento: sin credenciales no hay nada que mandar.")
        return 1
    a_quien = [args.a] if args.a else destinos
    if not a_quien:
        print("  No lo intento: no hay a quien.")
        return 1

    cuando = datetime.now().astimezone().isoformat(timespec="seconds")
    hubo_error = False
    for d in a_quien:
        try:
            es.enviar_alerta_email("prueba", "cam-de-prueba", 0.99, cuando, d)
            print("  OK  -> %s" % d)
        except Exception as e:
            hubo_error = True
            print("  FALLO -> %s : %s: %s" % (d, type(e).__name__, e))

    print()
    if hubo_error:
        print("  Algo no salio. Si dice 'Username and Password not accepted',")
        print("  la clave del .env no es una contrasena de aplicacion o esta")
        print("  mal copiada (van las 16 letras sin espacios).")
        return 1
    print("  Mandado. Fijate en la bandeja (y en el spam la primera vez).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
