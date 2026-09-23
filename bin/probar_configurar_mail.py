#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probar_configurar_mail.py - el configurador de mail, contra un Gmail de mentira.

No toca internet, no necesita credenciales y NO toca tu .env de verdad: todo
pasa en una carpeta temporal.

Lo que se prueba es lo que puede salir mal en serio:

  - que una clave que Gmail rechaza NO deje un .env escrito. Un archivo que
    parece configurado y no manda nada es peor que no tener archivo: el
    cartel del panel se apaga y el sistema sigue sin avisar.
  - que la clave no aparezca NUNCA en pantalla.
  - que los espacios con los que Google muestra la clave se saquen solos.
  - que cada falla diga cual de las tres cosas fue (clave, firewall, red) en
    vez de un error de smtplib.

    python bin/probar_configurar_mail.py
"""

from __future__ import annotations

import builtins
import contextlib
import io
import os
import shutil
import smtplib
import socket
import sys
import tempfile

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AQUI)

import configurar_mail as cm                              # noqa: E402

CLAVE_CON_ESPACIOS = "abcd efgh ijkl mnop"     # como la muestra Google
CLAVE = "abcdefghijklmnop"
REMITENTE = "horus@gmail.com"


class GmailDeMentira:
    """El unico Gmail que se toca acá."""
    modo = "ok"

    def __init__(self, *a, **k) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, usuario, clave):
        if GmailDeMentira.modo == "rechaza":
            raise smtplib.SMTPAuthenticationError(
                535, b"Username and Password not accepted")
        if GmailDeMentira.modo == "timeout":
            raise socket.timeout()
        if GmailDeMentira.modo == "sinred":
            raise socket.gaierror("Name or service not known")
        if clave != CLAVE:
            raise AssertionError("la clave llegó con espacios: %r" % clave)


class PreguntaDeMas(Exception):
    """El script pidió algo que el caso no previó.

    Vale la pena distinguirlo: significa que el flujo cambió, no que la
    lógica esté mal. Sin esto la prueba moría con un StopIteration pelado
    diez cuadros más abajo y había que leer el traceback para entender qué
    se rompió.
    """


def correr(respuestas, clave=CLAVE_CON_ESPACIOS):
    it = iter(respuestas)
    salida = io.StringIO()
    cm.smtplib.SMTP_SSL = GmailDeMentira
    viejo_in, viejo_pass = builtins.input, cm.getpass.getpass

    def _responder(p=""):
        print(p, file=salida)
        try:
            return next(it)
        except StopIteration:
            raise PreguntaDeMas(p.strip()) from None

    builtins.input = _responder
    cm.getpass.getpass = lambda p="": (print(p, file=salida), clave)[1]
    try:
        with contextlib.redirect_stdout(salida):
            cod = cm.main()
    except SystemExit as e:
        cod = e.code
    except PreguntaDeMas as e:
        cod = "preguntó de más: %r" % str(e)
    except Exception as e:                                 # noqa: BLE001
        cod = "%s: %s" % (type(e).__name__, str(e)[:80])
    finally:
        builtins.input, cm.getpass.getpass = viejo_in, viejo_pass
    return cod, salida.getvalue()


def limpiar():
    for f in (cm.ENV, cm.ENV + ".parcial"):
        if os.path.exists(f):
            os.remove(f)


def main() -> int:
    # El .env de verdad no se toca ni por accidente.
    tmp = tempfile.mkdtemp(prefix="horus_mail_")
    cm.FAST = tmp
    cm.ENV = os.path.join(tmp, ".env")
    sys.argv = ["configurar_mail.py"]

    fallas = []
    corridos = []

    def caso(nombre, ok, detalle=""):
        corridos.append(nombre)
        print(("  OK    " if ok else "  FALLA ") + "%-28s %s" % (nombre, detalle))
        if not ok:
            fallas.append(nombre)

    print("=" * 74)
    print("HORUS · configurador de mail — contra un Gmail de mentira")
    print("=" * 74)

    try:
        limpiar(); GmailDeMentira.modo = "ok"
        cod, out = correr([REMITENTE])
        hay = os.path.exists(cm.ENV)
        txt = open(cm.ENV, encoding="utf-8").read() if hay else ""
        caso("login ok escribe el .env",
             cod == 0 and hay and ("EMAIL_SENDER=%s" % REMITENTE) in txt,
             "codigo=%s" % cod)
        caso("guarda la clave sin espacios",
             "EMAIL_PASSWORD=%s\n" % CLAVE in txt)
        caso("no imprime la clave nunca",
             CLAVE not in out and CLAVE_CON_ESPACIOS not in out)
        caso("el archivo se llama .env", os.path.basename(cm.ENV) == ".env" and hay)

        limpiar(); GmailDeMentira.modo = "rechaza"
        cod, out = correr([REMITENTE])
        caso("clave rechazada: NO escribe nada",
             cod == 1 and not os.path.exists(cm.ENV))
        caso("y explica lo de app password",
             "contrasena de aplicacion" in out and "apppasswords" in out)

        limpiar(); GmailDeMentira.modo = "timeout"
        cod, out = correr([REMITENTE])
        caso("timeout: senala el firewall",
             cod == 1 and "bloqueado por el" in out and not os.path.exists(cm.ENV))

        limpiar(); GmailDeMentira.modo = "sinred"
        cod, out = correr([REMITENTE])
        caso("sin red: lo dice", cod == 1 and "No pude ni conectarme" in out)

        limpiar(); GmailDeMentira.modo = "ok"
        cod, out = correr([REMITENTE, "n"], clave="miclavenormal")
        caso("clave corta: avisa y frena",
             cod == 1 and "16" in out and not os.path.exists(cm.ENV))

        limpiar()
        cod, out = correr(["esto-no-es-un-mail"])
        caso("direccion sin arroba", cod == 1 and "no parece una direccion" in out)

        limpiar(); GmailDeMentira.modo = "ok"
        trampa = os.path.join(cm.FAST, ".env.txt")
        with open(trampa, "w", encoding="utf-8") as fh:
            fh.write("EMAIL_SENDER=x\n")
        try:
            cod, out = correr([REMITENTE])
            caso("detecta el .env.txt del Bloc", ".env.txt" in out)
        finally:
            os.remove(trampa)

        GmailDeMentira.modo = "ok"
        cod, out = correr(["n"])
        caso("respeta un .env que ya estaba",
             cod == 0 and "No toque nada" in out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 74)
    if fallas:
        print("%d de %d FALLA(S): %s"
              % (len(fallas), len(corridos), ", ".join(fallas)))
        return 1
    print("%d/%d pruebas OK" % (len(corridos), len(corridos)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
