
# coding: utf-8

# In[1]:

import gym_FPS
import gym
import random
from gym_FPS.utils import *
from time import *
import random
import numpy as np
env=gym.make('FPSDemo-v0')
env.set_env(client_DEBUG=False,env_DEBUG=False)
env.new_episode()
env.playerai()


# In[2]:

player_list=env.get_objid_list()
player_num=len(player_list)
comradeteam=[]
enemyteam=[]
searchteam_first=[]
searchteam_second=[]
alertteam=[]
towerteam=[]
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
search_num=3
comradeteam.sort()
listsample=[i for i in range(1,len(comradeteam))]
slice=random.sample(listsample,search_num)
print(slice)
comrade_search=[]
for i in range(search_num):
    comrade_search.append(comradeteam[slice[i]])
print(comrade_search)
dest_pos=[[-148.1,17.97,54.6],[-80.6,5.23,-84.7],[-80.6,5.23,-84.7]]
for i in range(search_num):
    env.move(dest_pos[i],objid_list=[comrade_search[i]])
again_destpos=[[-124.2,17.42,29.7],[-54.1,12.69,24.3]]
goned=[False for i in range(search_num-1)]
while True:
    for i in range(1,search_num):
        try:
            searchplayer=env.get_objid_list(name=0,pos=1)
            distance=np.array(searchplayer[comrade_search[i]])-np.array(dest_pos[i])
            gap=np.mat(distance)*np.mat(distance).T
            gap_double=np.sqrt(gap[0][0])
            if gap_double<6:
                env.move(again_destpos[i-1],objid_list=[comrade_search[i]])
                goned[i-1]=True
                break
        except:
            goned[i-1]=True
    if False not in goned:
        break
goned=[False for i in range(search_num-1)]
while True:
    for i in range(1,search_num):
        try:
            searchplayer=env.get_objid_list(name=0,pos=1)
            distance=np.array(searchplayer[comrade_search[i]])-np.array(again_destpos[i-1])
            gap=np.mat(distance)*np.mat(distance).T
            gap_double=np.sqrt(gap[0][0])
            if gap_double<6:
                goned[i-1]=True
                break
        except:
            goned[i-1]=True
    if False not in goned:
        break

time_after_search=1
curocean_pos= [-216.5,-0.70,-0.8]
for i in range(len(searchteam_first)):
    env.can_attack_move(objid_list=searchteam_first,destPos=curocean_pos)
    


        
#env.create_team(team2_leader,team_id4,2)


    
        
            
    
    



# In[3]:

print(searchplayer)


# In[4]:

rest_playerlist=env.get_objid_list()
rest_comradeteam=[]
for player_id,player_name in rest_playerlist.items():
    if player_name.find('队友')>=0:
        rest_comradeteam.append(player_id)
    elif player_id ==0:
        rest_comradeteam.append(player_id)
print(len(rest_comradeteam))
one_teamnum=7
second_teamnum=len(rest_comradeteam)-one_teamnum
rest_comradeteam.sort()
rest_listsample=[i for i in range(0,len(rest_comradeteam))]
rest_slice=random.sample(rest_listsample,one_teamnum)
one_attackteam=[]
second_attackteam=[]
for i in range(len(rest_comradeteam)):
    if i in rest_slice:
        one_attackteam.append(rest_comradeteam[i])
    else:
        second_attackteam.append(rest_comradeteam[i])


# In[5]:

one_attackteam.sort()
second_attackteam.sort()
print(second_attackteam)
if 0 in one_attackteam:
    env.create_team(0,one_attackteam,1)
    env.create_team(second_attackteam[0],second_attackteam,2)
else:
    env.create_team(one_attackteam[0],one_attackteam,1)
    env.create_team(0,second_attackteam,2)



# In[6]:

env.can_attack_move(objid_list=one_attackteam,destPos=dest_pos[0])
env.can_attack_move(objid_list=second_attackteam,destPos=dest_pos[1])
env.search_enemy_attack()
distgap=6
while True:
        rest_searchplayer=env.get_objid_list(name=0,pos=1)
        distance=np.array(rest_searchplayer[second_attackteam[0]])-np.array(dest_pos[1])
        gap=np.mat(distance)*np.mat(distance).T
        gap_double=np.sqrt(gap[0][0])
        if gap_double<distgap:
            env.can_attack_move(objid_list=second_attackteam,destPos=again_destpos[0])
            break


# In[7]:

rest_playerlist=env.get_objid_list()
rest_comradeteam=[]
for player_id,player_name in rest_playerlist.items():
    if player_name.find('队友')>=0:
        rest_comradeteam.append(player_id)
    elif player_id ==0:
        rest_comradeteam.append(player_id)
print(len(rest_comradeteam))
for player_id in rest_comradeteam:
    if player_id in one_attackteam:
        env.move(again_destpos[0],objid_list=[player_id])


# In[ ]:




# In[8]:

env.create_team(rest_comradeteam[0],rest_comradeteam,3)
env.can_attack_move(objid_list=rest_comradeteam,destPos=again_destpos[1])

