import socket
import struct
import numpy as np
import cv2
import config
import json

class GDClient:
    def __init__(self):
        self.host = config.BASE_URL
        self.cmd_port = config.COMMAND_PORT
        self.frame_port = config.FRAME_PORT
        self.cmd_sock = None
        self.frame_sock = None

    def connect(self):
        # Connect command socket
        self.cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.cmd_sock.connect((self.host, self.cmd_port))
        print(f"[CMD] Connected to {self.host}:{self.cmd_port}")

        # Connect frame socket
        self.frame_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.frame_sock.connect((self.host, self.frame_port))
        print(f"[FRAME] Connected to {self.host}:{self.frame_port}")

    def send_command(self, command):
        if not self.cmd_sock:
            raise Exception("Not connected to command server")
        self.cmd_sock.sendall(command.encode())

        response = self.cmd_sock.recv(1024).decode()
        data = json.loads(response)

        return data

    def receive_frame(self):
        if not self.frame_sock:
            raise Exception("Not connected to frame server")

        # Receive metadata: width + height
        meta_size = struct.calcsize('ii')
        meta_data = self._recv_exact(self.frame_sock, meta_size)
        if not meta_data:
            return None

        width, height = struct.unpack('ii', meta_data)
        image_size = width * height * 4

        image_data = self._recv_exact(self.frame_sock, image_size)
        if not image_data:
            return None

        img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))
        gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        gray_small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return gray_small

    def _recv_exact(self, sock, size):
        buffer = b''
        while len(buffer) < size:
            data = sock.recv(size - len(buffer))
            if not data:
                return None
            buffer += data
        return buffer

    def close(self):
        if self.cmd_sock:
            self.cmd_sock.close()
            print("[CMD] Connection closed")
        if self.frame_sock:
            self.frame_sock.close()
            print("[FRAME] Connection closed")
        self.cmd_sock = None
        self.frame_sock = None

gdclient = GDClient()