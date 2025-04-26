import socket

class GDClient:
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
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

if __name__ == "__main__":
    client = GDClient()
    client.connect()

    client.send_command("jump")
    client.send_command("restart")

    client.close()
