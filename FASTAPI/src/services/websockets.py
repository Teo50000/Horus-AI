from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, camara_config_id: int):
        await websocket.accept()
        self.active_connections.setdefault(camara_config_id, []).append(websocket)

    def disconnect(self, websocket: WebSocket, camara_config_id: int):
        connections = self.active_connections.get(camara_config_id)
        if not connections:
            return

        if websocket in connections:
            connections.remove(websocket)

        if not connections:
            self.active_connections.pop(camara_config_id, None)

    async def send_to_camera(self, message: str, camara_config_id: int):
        for connection in list(self.active_connections.get(camara_config_id, [])):
            try:
                await connection.send_text(message)
            except Exception:                            # noqa: BLE001
                self.disconnect(connection, camara_config_id)

    async def broadcast(self, message: str):
        """A todos los paneles abiertos.

        26/09: recorría la lista original y mandaba sin atrapar nada. Dos
        problemas, los dos del tipo que se ven como "se cayó el backend":

          - un panel que se cerró de golpe (pestaña matada, PC dormida) deja
            una conexión muerta que el servidor no se enteró de que murió. El
            `send_text` a esa conexión levanta, la excepción sube hasta el
            POST /alertas, que contesta 500 DESPUÉS de haber guardado la fila;
            el servicio reintenta, y así con cada alerta, para siempre;
          - si un panel se desconecta mientras se está recorriendo (el `await`
            le cede el turno), la lista cambia en medio del `for`.

        Ahora se recorre una copia y la conexión que falla se saca.
        """
        for cam_id, connections in list(self.active_connections.items()):
            for connection in list(connections):
                try:
                    await connection.send_text(message)
                except Exception:                        # noqa: BLE001
                    self.disconnect(connection, cam_id)


manager = ConnectionManager()
