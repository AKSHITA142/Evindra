import asyncio
from typing import Dict, List, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import logging

logger = logging.getLogger("datapilot.api.websocket_manager")


class ConnectionManager:
    """Manages active WebSocket connections per job_id and broadcasts real-time progress events."""

    def __init__(self):
        # Maps job_id -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accepts a WebSocket connection and subscribes it to a job_id channel."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket client connected to channel for job_id: {job_id}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Removes a WebSocket connection from a job_id channel."""
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket client disconnected from channel for job_id: {job_id}")

    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]):
        """Broadcasts a JSON event message to all clients subscribed to job_id."""
        if job_id not in self.active_connections:
            return

        dead_sockets: List[WebSocket] = []
        for connection in self.active_connections[job_id]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.warning(f"Error sending message over WebSocket to job {job_id}: {e}")
                dead_sockets.append(connection)

        for dead in dead_sockets:
            self.disconnect(dead, job_id)


    async def ping_heartbeat(self, job_id: str):
        """Sends a ping heartbeat message to active WebSocket connections to verify client health."""
        if job_id not in self.active_connections:
            return

        dead_sockets: List[WebSocket] = []
        for connection in list(self.active_connections[job_id]):
            try:
                await connection.send_json({"event": "ping", "type": "ping", "timestamp": asyncio.get_event_loop().time()})
            except Exception as e:
                logger.warning(f"WebSocket ping heartbeat failed for job {job_id}: {e}")
                dead_sockets.append(connection)

        for dead in dead_sockets:
            self.disconnect(dead, job_id)

    def cleanup_dead_connections(self, job_id: Optional[str] = None):
        """Removes stale or empty connection channels from active_connections registry."""
        if job_id:
            if job_id in self.active_connections and not self.active_connections[job_id]:
                del self.active_connections[job_id]
        else:
            empty_keys = [k for k, v in self.active_connections.items() if not v]
            for k in empty_keys:
                del self.active_connections[k]


# Global singleton instance
ws_manager = ConnectionManager()
