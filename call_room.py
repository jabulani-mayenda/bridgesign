import json
import threading

class CallRoomManager:
    def __init__(self):
        # Maps room_id -> list of active websocket connections (max 2)
        self.rooms = {}
        self.lock = threading.Lock()

    def join_room(self, room_id, ws):
        with self.lock:
            if room_id not in self.rooms:
                self.rooms[room_id] = []
            
            # Allow max 2 participants per room
            if len(self.rooms[room_id]) >= 2:
                try:
                    ws.send(json.dumps({"type": "error", "message": "Room is full"}))
                except Exception:
                    pass
                return False
                
            self.rooms[room_id].append(ws)
            print(f"[CallRoom] Client joined {room_id} ({len(self.rooms[room_id])}/2)")
            
            # When the second person joins, only the newcomer should create the
            # initial WebRTC offer. This avoids both peers offering at once.
            if len(self.rooms[room_id]) == 2:
                first_client, second_client = self.rooms[room_id]
                try:
                    first_client.send(json.dumps({"type": "ready", "should_offer": False}))
                except Exception:
                    pass
                try:
                    second_client.send(json.dumps({"type": "ready", "should_offer": True}))
                except Exception:
                    pass
            return True

    def leave_room(self, room_id, ws):
        with self.lock:
            if room_id in self.rooms:
                if ws in self.rooms[room_id]:
                    self.rooms[room_id].remove(ws)
                    print(f"[CallRoom] Client left {room_id} ({len(self.rooms[room_id])}/2)")
                
                # Notify remaining participant that peer left
                for client in self.rooms[room_id]:
                    try:
                        client.send(json.dumps({"type": "peer_left"}))
                    except Exception:
                        pass
                        
                if len(self.rooms[room_id]) == 0:
                    del self.rooms[room_id]

    def broadcast(self, room_id, sender_ws, message):
        """Relay message to the OTHER participant in the room."""
        with self.lock:
            if room_id not in self.rooms:
                return
            for client in self.rooms[room_id]:
                if client != sender_ws:
                    try:
                        client.send(message)
                    except Exception as e:
                        print(f"[CallRoom] Broadcast error: {e}")

# Global instance
room_manager = CallRoomManager()
