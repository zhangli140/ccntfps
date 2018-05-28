#coding=utf8

import numpy as np
import gym_FPS
import time
import gym
import threading
from gym_FPS.utils import *
from time import sleep
from copy import deepcopy
#outside_pos_list=[[125,-1,185],[185,-1,165],[205,-1,105],[185,-1,45],[125,-1,25],[55,-1,45],[45,-1,105],[55,-1,165]]
#inside_pos_list =[[125,-1,105],[125,-1,145],[165,-1,105],[125,-1,65],[85,-1,105]]
outside_pos_list=[[125,-1,175],[185,-1,155],[205,-1,95],[185,-1,35],[125,-1,15],[55,-1,35],[45,-1,95],[55,-1,155]]
inside_pos_list =[[125,-1,95],[125,-1,135],[165,-1,95],[125,-1,55],[85,-1,95]]
class Assignment(object):
    def __init__(self,env):
        self.outside_state = []
        self.inside_state = []
        self.outside_action = []
        self.inside_action = []
        self.thread_list = []
        self.outside_current_pos = {}
        self.inside_current_pos = {}
        self.outside_block_map = {}
        self.thread_list_len = 0
        self.env = env

    def outside_move_thread(self, uid, obj_pos):
        #print(uid)
        #road = []
        st = time.time()
        ed = time.time()
        pos = self.outside_block_map[uid]
        #road.append(pos)
        if (pos - obj_pos) % 8 >= 4:
            while pos != obj_pos:
                pos = (pos + 1) % 8
                #road.append(pos)
                self.env.move(outside_pos_list[pos], [uid], walkType='run')
                while get_distance(outside_pos_list[pos][0], outside_pos_list[pos][2], self.outside_current_pos[uid][0],
                                   self.outside_current_pos[uid][1]) > 6 and ed - st < 180:
                    sleep(0.1)
                    ed = time.time()
        else:
            while pos != obj_pos:
                pos = (pos + 7) % 8
                #road.append(pos)
                self.env.move(outside_pos_list[pos], [uid], walkType='run')
                while get_distance(outside_pos_list[pos][0], outside_pos_list[pos][2], self.outside_current_pos[uid][0],
                                   self.outside_current_pos[uid][1]) > 6 and ed - st < 180:
                    sleep(0.1)
                    ed = time.time()
        #print('out(id,pos):', uid, obj_pos, road)
        self.thread_list_len -= 1

    def inside_move_thread(self, uid, obj_pos):
        #print(uid)
        st = time.time()
        ed = time.time()
        self.env.move(inside_pos_list[obj_pos],[uid], walkType='run')
        while get_distance(inside_pos_list[obj_pos][0], inside_pos_list[obj_pos][2], self.inside_current_pos[uid][0],
                           self.inside_current_pos[uid][1]) > 6 and ed - st < 180:
            sleep(0.1)
            ed = time.time()
        #print('in(id,pos):', uid, obj_pos)
        self.thread_list_len -= 1

    def outside_agent_choose(self, obj_pos):
        outside_choose_rank = [(obj_pos+7) % 8, (obj_pos+1) % 8, (obj_pos+6) % 8, (obj_pos+2) % 8, (obj_pos+5) % 8,
                               (obj_pos+3) % 8, (obj_pos+4) % 8]
        for point in outside_choose_rank:
            while len(self.outside_state[point]) > 0 and self.outside_action[obj_pos] > 0:
                self.env.sup_outside[int(obj_pos/2)].append(self.outside_state[point][-1:])
                self.thread_list.append(threading.Thread(target=self.outside_move_thread, args=(self.outside_state[point].pop(), obj_pos)))
                self.outside_action[obj_pos] -= 1

    def inside_agent_choose(self, obj_pos):
        
        inside_choose_rank = []
        if obj_pos == 0:
            inside_choose_rank = [5, 1, 2, 3, 4]
        else:
            inside_choose_rank = [5, 0, (obj_pos+2) % 4+1, obj_pos % 4+1, (obj_pos+3) % 4+1]
        for point in inside_choose_rank:
            while len(self.inside_state[point]) > 0 and self.inside_action[obj_pos] > 0:
                self.env.sup_inside[obj_pos].append(self.inside_state[point][-1:])

                self.thread_list.append(threading.Thread(target=self.inside_move_thread, args=(self.inside_state[point].pop(), obj_pos)))
                self.inside_action[obj_pos] -= 1
 
    def make_state(self):
        self.env._make_feature()
        self.outside_state = [[],[],[],[],[],[],[],[]]
        self.outside_block_map = {}
        pos = outside_pos_list
        for uid,ut in self.env.state['units_myself'].items():
            if pos[0][0]-10<=ut['POSITION'][0] and pos[0][0]+10>=ut['POSITION'][0] and pos[0][2]-10<=ut['POSITION'][2] and pos[0][2]+10>=ut['POSITION'][2]:
                self.outside_state[0].append(uid)
                self.outside_block_map[uid] = 0
            elif pos[0][0]+10<ut['POSITION'][0] and pos[2][2]+10<ut['POSITION'][2]:
                self.outside_state[1].append(uid)
                self.outside_block_map[uid] = 1
            elif pos[2][0]-10<=ut['POSITION'][0] and pos[2][0]+10>=ut['POSITION'][0] and pos[2][2]-10<=ut['POSITION'][2] and pos[2][2]+10>=ut['POSITION'][2]:
                self.outside_state[2].append(uid)
                self.outside_block_map[uid] = 2
            elif pos[4][0]+10<ut['POSITION'][0] and pos[2][2]-10>ut['POSITION'][2]:
                self.outside_state[3].append(uid)
                self.outside_block_map[uid] = 3
            elif pos[4][0]-10<=ut['POSITION'][0] and pos[4][0]+10>=ut['POSITION'][0] and pos[4][2]-10<=ut['POSITION'][2] and pos[4][2]+10>=ut['POSITION'][2]:
                self.outside_state[4].append(uid)
                self.outside_block_map[uid] = 4
            elif pos[4][0]-10>ut['POSITION'][0] and pos[6][2]-10>ut['POSITION'][2]:
                self.outside_state[5].append(uid)
                self.outside_block_map[uid] = 5
            elif pos[6][0]-10<=ut['POSITION'][0] and pos[6][0]+10>=ut['POSITION'][0] and pos[6][2]-10<=ut['POSITION'][2] and pos[6][2]+10>=ut['POSITION'][2]:
                self.outside_state[6].append(uid)
                self.outside_block_map[uid] = 6
            elif pos[0][0]-10>ut['POSITION'][0] and pos[6][2]+10<ut['POSITION'][2]:
                self.outside_state[7].append(uid)
                self.outside_block_map[uid] = 7

        self.inside_state = [[],[],[],[],[],[]]
        pos = inside_pos_list
        for uid,ut in self.env.state['units_enemy'].items():
            if pos[0][0]-10<=ut['POSITION'][0] and pos[0][0]+10>=ut['POSITION'][0] and pos[0][2]-10<=ut['POSITION'][2] and pos[0][2]+10>=ut['POSITION'][2]:
                self.inside_state[0].append(uid)
            elif pos[1][0]-10<=ut['POSITION'][0] and pos[1][0]+10>=ut['POSITION'][0] and pos[1][2]-10<=ut['POSITION'][2] and pos[1][2]+10>=ut['POSITION'][2]:
                self.inside_state[1].append(uid)
            elif pos[2][0]-10<=ut['POSITION'][0] and pos[2][0]+10>=ut['POSITION'][0] and pos[2][2]-10<=ut['POSITION'][2] and pos[2][2]+10>=ut['POSITION'][2]:
                self.inside_state[2].append(uid)
            elif pos[3][0]-10<=ut['POSITION'][0] and pos[3][0]+10>=ut['POSITION'][0] and pos[3][2]-10<=ut['POSITION'][2] and pos[3][2]+10>=ut['POSITION'][2]:
                self.inside_state[3].append(uid)
            elif pos[4][0]-10<=ut['POSITION'][0] and pos[4][0]+10>=ut['POSITION'][0] and pos[4][2]-10<=ut['POSITION'][2] and pos[4][2]+10>=ut['POSITION'][2]:
                self.inside_state[4].append(uid)
            else:
                self.inside_state[5].append(uid)

    def current_pos_thread(self):

        self.outside_current_pos = {}
        self.inside_current_pos = {}

        while not self.env.stop:
            self.env._make_feature()
            for uid, ut in self.env.state['units_myself'].items():
                self.outside_current_pos[uid] = [ut['POSITION'][0], ut['POSITION'][2]]
            for uid, ut in self.env.state['units_enemy'].items():
                self.inside_current_pos[uid] = [ut['POSITION'][0], ut['POSITION'][2]]
            sleep(0.1)

    def Assign(self, outside_a, inside_a):
        self.env.stop = False
        self.outside_action = [outside_a[0], 0, outside_a[1], 0, outside_a[2], 0, outside_a[3], 0]
        self.inside_action = deepcopy(inside_a)
        self.thread_list = []
        self.make_state()
        #print('os: ',self.outside_state)
        #print('is: ', self.inside_statse)
        self.env.sup_outside = [[],[],[],[]]
        self.env.sup_inside = [[],[],[],[],[]]
        # -------outside-------
        idx_list = [0,1,2,3]
        while len(idx_list) > 0:
            len_array = np.array([len(self.outside_state[0]),len(self.outside_state[2]),len(self.outside_state[4]),len(self.outside_state[6])])
            outside_rank = np.array([self.outside_action[0],self.outside_action[2],self.outside_action[4],self.outside_action[6]]) - len_array
            idx = idx_list[np.argmax(outside_rank[idx_list])]
            idx_list.remove(idx)
            idx *= 2
            while self.outside_action[idx]>0 and len(self.outside_state[idx])>0:
                self.env.move(outside_pos_list[idx],[self.outside_state[idx].pop()], walkType='run')
                self.outside_action[idx] -= 1
            if self.outside_action[idx]>0:
                self.outside_agent_choose(idx)
        #--------inside--------
        idx_list = [0,1,2,3,4]
        while len(idx_list) > 0:
            inside_rank = np.array(self.inside_action) - np.array([len(self.inside_state[0]),len(self.inside_state[1]),len(self.inside_state[2]),len(self.inside_state[3]),len(self.inside_state[4])])
            idx = idx_list[np.argmax(inside_rank[idx_list])]
            idx_list.remove(idx)
            while self.inside_action[idx]>0 and len(self.inside_state[idx])>0:
                self.env.move(inside_pos_list[idx],[self.inside_state[idx].pop()], walkType='run')
                self.inside_action[idx] -= 1
            if self.inside_action[idx]>0:
                self.inside_agent_choose(idx)
        
        self.thread_list_len = len(self.thread_list)
        #print('tl: ', self.thread_list_len)
        #current_pos_thread
        thread_env = threading.Thread(target=self.current_pos_thread)
        thread_env.setDaemon(True)
        thread_env.start()
        sleep(0.1)
        #Multi-Thread
        for th in self.thread_list:
            th.setDaemon(True)
            th.start()
        while self.thread_list_len > 0 and self.env.stop is False:
            #print(self.thread_list_len)
            sleep(0.1)
            if self.thread_list_len == 0:
                self.env.stop = True