#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
configurar_mail.py - dejar el aviso por mail andando, de una sentada.

Por que existe un script para escribir un archivo de dos lineas: porque ese
archivo en Windows es sorprendentemente dificil de crear bien.

  - El Bloc de notas le agrega .txt sin avisar. Queda ".env.txt", que no lo
    lee nadie, y el sistema sigue sin mandar mails exactamente igual que
    antes. El Explorador de Windows esconde la extension, asi que en pantalla
    se ve ".env" igual.
  - Google muestra la clave de aplicacion en cuatro grupos de cuatro letras.
    Copiada con los espacios no entra.
  - Y si algo de eso sale mal, el error que tira Gmail no dice cual de las
    cosas fue.

Este script pide los dos datos, escribe el archivo bien, y despues PRUEBA el
login de verdad contra Gmail antes de decir que quedo listo.

La contrasena se pide sin eco, no se imprime en ningun momento, no queda en
el historial de la consola y no sale de tu maquina salvo hacia Gmail. Yo no
la veo ni la necesito.

    python bin\\configurar_mail.py
    python bin\\configurar_mail.py --revisar     (no escribe nada, solo mira)
"""

from __future__ import annotations

import argparse
import getpass
import os
import smtplib
import socket
import ssl
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAST = os.path.join(RAIZ, "FASTAPI")
ENV = os.path.join(FAST, ".env")
APP_PASSWORDS = "https://myaccount.google.com/apppasswords"


def _linea() -> None:
    print("-" * 62)


def buscar_env_mal_guardado() -> list:
    """Los nombres con los que termina un .env hecho a mano en Windows."""
    sospechosos = []
    if os.path.isdir(FAST):
        for f in os.listdir(FAST):
            b = f.lower()
            if b != ".env" and (b.startswith(".env.") and b != ".env.example"
                                or b in ("env", "env.txt", "_env")):
                sospechosos.append(f)
    return sospechosos


def revisar() -> bool:
    """Que hay hoy. No toca nada."""
    print("Estado actual")
    _linea()
    print("  archivo  : %s" % ENV)
    if os.path.exists(ENV):
        with open(ENV, encoding="utf-8", errors="replace") as fh:
            claves = [l.split("=")[0].strip() for l in fh
                      if l.strip() and not l.startswith("#") and "=" in l]
        print("  existe   : si, con %s" % (", ".join(claves) or "nada adentro"))
    else:
        print("  existe   : NO")

    malos = buscar_env_mal_guardado()
    if malos:
        print()
        print("  OJO: hay archivos con nombre parecido que el backend NO lee:")
        for m in malos:
            print("     %s" % m)
        print("  El Bloc de notas agrega .txt solo. El archivo tiene que")
        print("  llamarse .env a secas.")
    return os.path.exists(ENV)


def pedir_datos():
    print()
    print("Te voy a pedir dos cosas.")
    _linea()
    print("1) El Gmail desde el que salen las alertas.")
    print("   Puede ser el tuyo, pero conviene uno aparte para Horus.")
    print()
    remitente = input("   Direccion: ").strip()
    if "@" not in remitente:
        print("   Eso no parece una direccion de mail.")
        return None, None
    if not remitente.lower().endswith("@gmail.com"):
        print("   (No es un gmail. Va a andar igual si tu proveedor acepta")
        print("    SMTP con contrasena, pero los mensajes de ayuda de abajo")
        print("    estan escritos para Gmail.)")

    print()
    print("2) La CONTRASENA DE APLICACION. No es la clave con la que entras")
    print("   a tu Gmail: es una de 16 letras que genera Google aparte, y")
    print("   solo sirve para esto. Se saca en:")
    print("       %s" % APP_PASSWORDS)
    print("   Hace falta tener la verificacion en dos pasos activada; si no,")
    print("   esa pagina no te deja generar ninguna.")
    print()
    print("   Pegala y dale Enter. No se va a ver mientras escribis, y no la")
    print("   imprimo en ningun lado.")
    print()
    try:
        clave = getpass.getpass("   Clave de aplicacion: ")
    except (KeyboardInterrupt, EOFError):
        print("\n   Cancelado.")
        return None, None

    # Google la muestra como "abcd efgh ijkl mnop". Con los espacios no entra.
    limpia = clave.replace(" ", "").replace("\t", "").strip()
    if not limpia:
        print("   No escribiste nada.")
        return None, None
    if len(limpia) != 16:
        print()
        print("   Ojo: las contrasenas de aplicacion de Google tienen 16")
        print("   letras y esta tiene %d. Si pusiste la clave normal de tu"
              % len(limpia))
        print("   cuenta, Gmail la va a rechazar.")
        print()
        if (input("   Sigo igual? [s/N]: ").strip().lower() or "n") != "s":
            return None, None
    return remitente, limpia


def probar_login(remitente: str, clave: str, host: str = "smtp.gmail.com",
                 puerto: int = 465) -> bool:
    """Conectar y autenticar, sin mandar nada.

    Se prueba ANTES de dar por buena la configuracion. Escribir el archivo y
    decir "listo" sin haber autenticado una sola vez es como dar por instalado
    un modelo porque el archivo existe.
    """
    print()
    print("   Probando el login contra %s ..." % host)
    try:
        with smtplib.SMTP_SSL(host, puerto, timeout=15,
                              context=ssl.create_default_context()) as s:
            s.login(remitente, clave)
        print("   OK. Gmail acepto la clave.")
        return True
    except smtplib.SMTPAuthenticationError as e:
        codigo = getattr(e, "smtp_code", "?")
        print("   RECHAZADA (%s)." % codigo)
        print()
        print("   Las tres causas, por orden de frecuencia:")
        print("     1. No es una contrasena de aplicacion sino la clave")
        print("        normal de la cuenta. Gmail no acepta esa por SMTP.")
        print("     2. Esta mal copiada. Son 16 letras; los espacios que")
        print("        muestra Google no van (esos los saco yo solo).")
        print("     3. La direccion no es la misma cuenta donde generaste")
        print("        la clave.")
        print()
        print("   Se generan en: %s" % APP_PASSWORDS)
        return False
    except (socket.timeout, TimeoutError):
        print("   No contesto a tiempo. El puerto %d esta bloqueado por el" % puerto)
        print("   firewall, el antivirus o la red donde estas.")
        return False
    except (socket.gaierror, OSError) as e:
        print("   No pude ni conectarme: %s" % e)
        print("   Revisa que tengas internet y que el %d no este filtrado."
              % puerto)
        return False
    except Exception as e:                                   # noqa: BLE001
        print("   Fallo raro: %s: %s" % (type(e).__name__, str(e)[:160]))
        return False


def escribir_env(remitente: str, clave: str) -> None:
    """El archivo, con el nombre bien puesto.

    Se escribe desde python justamente para que se llame `.env` y nada mas.
    """
    contenido = (
        "# Generado por bin/configurar_mail.py\n"
        "# Este archivo NO se sube al repositorio (esta en .gitignore).\n"
        "# La clave de aplicacion solo sirve para mandar mails desde Horus:\n"
        "# si alguna vez se filtra, se revoca desde la cuenta de Google y\n"
        "# se genera otra, sin tocar la contrasena de la cuenta.\n"
        "EMAIL_SENDER=%s\n"
        "EMAIL_PASSWORD=%s\n" % (remitente, clave))
    tmp = ENV + ".parcial"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(contenido)
    os.replace(tmp, ENV)
    try:
        os.chmod(ENV, 0o600)          # en Windows no hace mucho, no molesta
    except OSError:
        pass


def destinos_cargados() -> list:
    """A quien se le avisa hoy, segun la base del backend."""
    try:
        sys.path.insert(0, FAST)
        anterior = os.getcwd()
        os.chdir(FAST)
        try:
            from sqlmodel import Session, select
            from src.database import engine
            from src.models.camara_model import NumeroEmergencia
            with Session(engine) as s:
                return [d for d in (c.direccion_mail()
                                    for c in s.exec(select(NumeroEmergencia)).all())
                        if d]
        finally:
            os.chdir(anterior)
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="Configurar el aviso por mail.")
    ap.add_argument("--revisar", action="store_true",
                    help="solo mirar que hay, sin escribir nada")
    args = ap.parse_args()

    print("=" * 62)
    print("HORUS - dejar andando el aviso por mail")
    print("=" * 62)
    print()
    print("Las alertas ya se guardan y se ven en el panel. Lo que falta es")
    print("que salgan de la casa: que a alguien le llegue un mail cuando")
    print("pasa algo y no este mirando la pantalla.")
    print()

    revisar()
    if args.revisar:
        return 0

    if os.path.exists(ENV):
        print()
        if (input("Ya hay un .env. Lo reemplazo? [s/N]: ").strip().lower()
                or "n") != "s":
            print("No toque nada.")
            return 0

    remitente, clave = pedir_datos()
    if not remitente:
        return 1

    if not probar_login(remitente, clave):
        print()
        print("No escribo el .env con una clave que Gmail rechaza: quedaria")
        print("un archivo que parece configurado y no manda nada. Arregla lo")
        print("de arriba y corre esto de nuevo.")
        return 1

    escribir_env(remitente, clave)
    del clave                          # no queda dando vueltas mas de lo justo
    print()
    print("Escrito: %s" % ENV)

    malos = buscar_env_mal_guardado()
    if malos:
        print()
        print("Podes borrar estos, que no los lee nadie: %s" % ", ".join(malos))

    print()
    _linea()
    destinos = destinos_cargados()
    if destinos:
        print("Se les va a avisar a: %s" % ", ".join(destinos))
    else:
        print("OJO: no hay ningun contacto con direccion de mail cargada.")
        print("El mail ya puede salir, pero no hay a quien mandarselo.")
        print("Se agregan en el panel, en Ajustes.")

    print()
    print("Para confirmarlo de punta a punta:")
    print("    python bin\\probar_mail.py --mandar")
    print()
    print("Y despues reinicia el backend: las credenciales se leen al")
    print("arrancar, asi que el que este corriendo ahora todavia no las tiene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
