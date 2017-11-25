# coding: utf-8

import gym
import gym_FPS
import time
from gym_FPS.utils import *
from gym_FPS.client import Client
from tqdm import *
import random
import threading
import socket

def receive():
    client = Client(SERVERPORT=5144, DEBUG=False)
    #client=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    #client.connect(('127.0.0.1',5144))

    global action
    global flag
    while True:
        action = int(client.receive())
        print(action)
        flag = True

env = gym.make('FPSDemo-v3')
env.set_env(client_DEBUG=False,env_DEBUG=False)
obs = env.reset()

action = 0
flag = False
count = 0

t = threading.Thread(target=receive)

t.start()

while True:
    if count % 100 == 0:
        if action != 4 or count == 0:
            print(action)
            env.step(action)
    time.sleep(0.01)
    count += 1
    if flag:
        count = 0
        flag = False
