import socket
import config

class GDClient:
    def __init__(self):
        self.host = config.BASE_URL
        self.port = config.CLIENT_PORT
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")

    def send_command(self, command):
        if not self.sock:
            raise Exception("Not connected to server")
        self.sock.sendall(command.encode())

        response = self.sock.recv(1024)
        print(f"Server response: {response.decode()}")

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
            print("Connection closed")