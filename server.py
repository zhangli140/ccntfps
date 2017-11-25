import socket
import base64

class Server():
    def __init__(self, SEVERIP='127.0.0.1', SERVERPORT=5144): 
        self.s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.s.bind((SEVERIP, SERVERPORT))
        self.s.listen(1)
        self.server,addr = self.s.accept()

    def send(self, mes):
        self.server.sendall(base64.b64encode(bytes(mes, encoding='utf8'))+b'\n')


server=Server()
while True:
    a = input()
    server.send(a)