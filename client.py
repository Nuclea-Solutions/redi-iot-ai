import socket
import json

ip, port = "127.0.0.1", 8080

def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((ip, port))

    while True:
        data = client.recv(1024)
        if not data:
            continue

        payload = json.loads(data.decode())
        print(payload)


main()