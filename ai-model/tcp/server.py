import socket
import struct
import numpy as np
import cv2
import config

class GDServer:
    def __init__(self):
        self.host = config.BASE_URL
        self.port = config.SERVER_PORT
        self.sock = None
        self.conn = None

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

        self.conn, addr = self.sock.accept()
        print(f"Connection accepted from {addr}")

    def receive_frame(self):
        # 1. First receive metadata: width and height
        meta_size = struct.calcsize('II')  # 2 unsigned ints
        meta_data = self._recv_exact(meta_size)
        if not meta_data:
            return None

        width, height = struct.unpack('II', meta_data)

        # 2. Then receive the full image buffer
        image_size = width * height * 4  # RGBA
        image_data = self._recv_exact(image_size)
        if not image_data:
            return None

        # 3. Convert to numpy array
        img = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 4))

        # Optional: Convert from RGBA to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img_bgr

    def _recv_exact(self, size):
        """ Helper to receive exactly 'size' bytes """
        buffer = b''
        while len(buffer) < size:
            data = self.conn.recv(size - len(buffer))
            if not data:
                return None
            buffer += data
        return buffer

    def close(self):
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
        print("Server closed")

