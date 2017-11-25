# coding: utf-8

import time  
import socket
from .utils import *
import base64

class Client(object):
    def __init__(self, SEVERIP='127.0.0.1', SERVERPORT=5123, DEBUG=True):
        self.SEVERIP = SEVERIP
        self.SERVERPORT = SERVERPORT
        self.sock = self.connect()
        self.DEBUG = DEBUG
        self.receive_buf = ''
        #self.units = dict()
        #self.states = dict()
        #self.episode_id = 0
        self.notify = []

    def connect(self, ):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.sock.connect((self.SEVERIP,self.SERVERPORT))
        #self.sock.settimeout(1000)
        return self.sock

    def send(self, cmd, ):
        '''
        le=0
        count=0
        while True:
            sock.send(bytes(cmd[le:], encoding='utf-8'))
            if self.DEBUG:
                count+=1
                print('send:%d, len:%d, str:%s' ,count, le, cmd[:le])
            if le==0:
                break
        '''
        try:
            self.sock.sendall(base64.b64encode(bytes(cmd, encoding='utf8'))+b'\n')
            if self.DEBUG:
                print('send over:%s'%cmd)
        except:
            print('send failed!!!!!!!!!!!!!!!!!!')

    def receive(self, ):
        try:
            receive_buf = ''
            flag = True
            count = 0
            while flag:
                st = self.sock.recv(8192)
                count += 1
                if 10 == st[-1]:
                    flag = False
                st=base64.b64decode(st).decode()
                receive_buf += st
            if receive_buf[-1] == '`':
                receive_buf = receive_buf[:-1]
            if self.DEBUG:
                print('receive:',receive_buf)
                print(count)

            if receive_buf.find('notify')>-1:
                self.notify.append(receive_buf)
                return self.receive()
            return receive_buf
        except:
            print('received time out!')


