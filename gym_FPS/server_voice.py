import time  
import socket
import base64
import threading

class Server(object):
    def __init__(self, SEVERIP='127.0.0.1', SERVERPORT=8338, DEBUG=True):
        self.SEVERIP = SEVERIP
        self.SERVERPORT = SERVERPORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((self.SEVERIP, self.SERVERPORT))
        self.sock.listen(5)
        self.DEBUG = DEBUG
        self.t1 = threading.Thread(target=self._wait_for_connect)
        self.t1.start()
        self.t2 = None
        self.buff = ''

    def _wait_for_connect(self,):
        while True:
            self.c, self.addr = self.sock.accept()
            try:
                while True:
                    data = self._receive()
                    if len(data) < 1:
                        break
                    self.buff += data.decode('gbk')
                    print(self.buff)
            except Exception as e:
                print(e)

    def _receive(self, ):
        try:
            data = self.c.recv(4096)
            if len(data) > 0:
                if self.DEBUG:
                    print('recv', data.decode('gbk'))
                return data
        except:
            print('received time out!')