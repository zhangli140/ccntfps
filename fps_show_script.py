# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 18:37:04 2017

@author: ZhangLi85
"""

import gym_FPS
import gym
import random
import threading
from gym_FPS.utils import *
from gym_FPS.control import *
from time import *
import numpy as np

env=gym.make('FPSDemo-v0')
env.set_env(client_DEBUG=False,env_DEBUG=False)
env.new_episode()
env.playerai()

#场景一：派出侦察员，并消灭敌方哨队
player_list=env.get_objid_list()
player_num=len(player_list)
comradeteam=[]
enemyteam=[]
searchteam_first=[]
searchteam_second=[]
alertteam=[]
towerteam=[]

##敌我分队初始化
for player_id,player_name in player_list.items():
    if player_name.find('队友')>=0:
        comradeteam.append(player_id)
    elif player_id ==0:
        comradeteam.append(player_id)
    elif player_name.find('巡逻')>=0:
        if int(player_name[-1])<=3:
            searchteam_first.append(player_id)
        else:
            searchteam_second.append(player_id)
    elif player_name.find('驻守')>=0:
        alertteam.append(player_id)
    else:
        towerteam.append(player_id)

print(len(comradeteam))
print(len(searchteam_first))
print(len(searchteam_second))
print(len(alertteam))
print(len(towerteam))

##侦察员出发
search_num=3
comradeteam.sort()
listsample=[i for i in range(1,len(comradeteam))]
slice=random.sample(listsample,search_num)
print(slice)
#侦察队组建！
comrade_search=[]
for i in range(search_num):
    comrade_search.append(comradeteam[slice[i]])
print(comrade_search)
#侦察队目标
dest_pos0=[[-148.1,17.97,54.6],[-80.6,5.23,-84.7],[-80.6,5.23,-84.7]]
dest_pos=[[-208.1,-1,98.6],[-90.6,-1,-84.7],[-90.6,-1,-84.7]]
curocean_pos= [-216.5,-0.70,-0.8]
dest_pos_t1=[[-208.1,-1,98.6],[-178.6,-1,119.7],[-154.1,-1,48.9],\
             [-100.6,-1,80.1],[-58.6,-1,28.1]]
#dest_pos_t2=[[-180.5,-1,-70.4],[-168.5,-1,0.4],[-154.1,-1,48.9],[-58.6,12.68,28.1]]
dest_pos_t2=[[-180.5,-1,-70.4],[-154.1,-1,48.9],\
             [-100.6,-1,80.1],[-58.6,-1,28.1]]

scout_team=["team1","team2","team2"]
scout_follow=[dest_pos_t1, dest_pos_t2, dest_pos_t2]

#侦察队出发
for i in range(search_num):
    sct = Scout(comrade_search[i], env, "Pioneer", scout_team[i], [dest_pos[i]], \
                scout_follow[i])
    task_s = threading.Thread(target=sct.run, args=())
    task_s.start()
#    env.move(dest_pos[i],objid_list=[comrade_search[i]],walkType='run')

goned=[False for i in range(search_num-1)]
while True:
    for i in range(1,search_num):
        try:
            searchplayer=env.get_objid_list(name=0,pos=1)
            distance=np.array(searchplayer[comrade_search[i]])-np.array(curocean_pos)
            gap=np.mat(distance)*np.mat(distance).T
            gap_double=np.sqrt(gap[0][0])
            if gap_double>6:
#                env.move(again_destpos[i-1],objid_list=[comrade_search[i]])
                goned[i-1]=True
                break
        except:
            goned[i-1]=True
    if False not in goned:
        break

goned=[False for i in range(search_num-1)]
time_after_search=1

#分头行动！老大带人去打老窝，小队负责阻击敌方哨队！
one_attackteam=[]#主队
second_attackteam=[]

rest_playerlist=env.get_objid_list()
for player_id,player_name in rest_playerlist.items():
    if player_id not in comrade_search:
        if player_name.find('队友')>=0:
            if len(second_attackteam) < 5:
                second_attackteam.append(player_id)
            else:
                one_attackteam.append(player_id)
        elif player_id ==0:
            one_attackteam.append(player_id)
print("team1:",len(one_attackteam),", team2:",len(second_attackteam),", our_scouts:",len(comrade_search))

one_attackteam.sort()
second_attackteam.sort()
print(one_attackteam)
print(second_attackteam)

env.register("wait_team1",0)
env.register("wait_team2",0)
env.register("team1", None)
env.register("team2", None)
env.register("ready_team1", False)
env.register("ready_team2", False)
env.register("clear1",False)
env.register("clear2",False)
players = env.get_objid_list()
poses = env.get_objid_list(name=0,pos=1)
env.register("players",players)
env.register("poses",poses)

if 0 in one_attackteam:
    env.create_team(0,one_attackteam,2)
    env.create_team(second_attackteam[0],second_attackteam,1)
else:
    env.create_team(one_attackteam[0],one_attackteam,1)
    env.create_team(0,second_attackteam,2)

#第一战！主队去迂回，分队挡住
task_t1 = TeamWork("team1",env, one_attackteam, dest_pos_t1)
task_thread1 = threading.Thread(target=task_t1.run, args=())
task_thread1.start()
sleep(3)

task_t2 = TeamWork("team2",env, second_attackteam, dest_pos_t2)
task_thread2 = threading.Thread(target=task_t2.run, args=())
task_thread2.start()

#env.register("team1t", task_thread1)
#env.register("team2t", task_thread2)
#env.can_attack_move(objid_list=second_attackteam,destPos=dest_pos[1])
#被敌方发现了！
for i in range(len(searchteam_first)):
    env.can_attack_move(objid_list=searchteam_first,destPos=curocean_pos)
#env.can_attack_move(objid_list=one_attackteam,destPos=dest_pos[0])    
#env.search_enemy_attack()
reader = Reader(env)
reader_task = threading.Thread(target=reader.run, args=())
reader_task.start()

moniter = Moniter(env)
moniter_task = threading.Thread(target=moniter.run, args=())
moniter_task.start()