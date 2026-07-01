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
        for connection in self.active_connections.get(camara_config_id, []):
            await connection.send_text(message)

    async def broadcast(self, message: str):
        for connections in self.active_connections.values():
            for connection in connections:
                await connection.send_text(message)


manager = ConnectionManager()