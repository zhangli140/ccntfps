import numpy as np
from random import sample, randint, random, shuffle
import pickle as pickle
class ReplayMemory_with_dead_index:
    def __init__(self, capacity,resolution):
        channels = 1
        state_shape = (capacity, resolution[0], resolution[1], channels)
        self.s1 = np.zeros(state_shape, dtype=np.float32)
        self.s2 = np.zeros(state_shape, dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int32)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.command_size = np.zeros(capacity, dtype=np.int32)
        self.command_size_next = np.zeros(capacity, dtype=np.int32)
        self.isterminal = np.zeros(capacity, dtype=np.int32)
        self.dead_e_index = {}


        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.sample_pos = 0

    def add_transition(self, s1, action, s2, isterminal, reward, command_size,command_size_next, dead_e_index):
        self.s1[self.pos, :, :, 0] = s1
        self.a[self.pos] = action
        if not isterminal:
            self.s2[self.pos, :, :, 0] = s2
        self.isterminal[self.pos] = isterminal
        self.r[self.pos] = reward
        self.command_size[self.pos] = command_size
        self.command_size_next[self.pos] = command_size_next
        self.dead_e_index[self.pos] = dead_e_index

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def clear(self):
        self.size = 0
        self.pos = 0
        self.sample_pos = 0
    def shuffle(self):
        self.sample_pos = 0
        self.index = np.array(range(0,self.size))
        shuffle(self.index)
    def store(self,path):
        stored_mem = {}
        stored_mem['s1'] = self.s1
        stored_mem['s2'] = self.s2
        stored_mem['a'] = self.a
        stored_mem['r'] = self.r
        stored_mem['command_size'] = self.command_size
        stored_mem['command_size_next'] = self.command_size_next
        stored_mem['isterminal'] = self.isterminal
        stored_mem['dead_e_index'] = self.dead_e_index
        stored_mem['size'] = self.size
        stored_mem['pos'] = self.pos
        # print("self.a:{}".format(self.a[0:10]))
        f1 = open(path, "wb")
        pickle.dump(stored_mem, f1)
        f1.close()

    def load(self,path):
        f1 = open(path,"rb")
        stored_mem = pickle.load(f1)
        # print(np.array(stored_mem['s1']).shape)
        self.s1[0:len(stored_mem['s1']),:,:,:]= stored_mem['s1']
        self.a[0:len(stored_mem['a'])] = stored_mem['a']
        self.pos = stored_mem['size']
        self.size = stored_mem['pos']
        # print("test:self.a:{}".format(self.a[0:10]))
        f1.close()

    def get_sample(self, sample_size):
        # 1/3 validation
        index = range(self.sample_pos,min(self.sample_pos+sample_size,self.size*2//3))
        # print("sample_pos:{}".format(self.sample_pos))
        # print("self.size:{}".format(self.size))
        self.sample_pos = min(self.sample_pos+sample_size,self.size*2//3)
        s1_list = []
        a_list = []
        s2_list = []
        isterminal_list = []
        r_list = []
        command_size_list = []
        command_size_next_list = []
        dead_e_index_list = []
        for i in index:
            s1_list.append(self.s1[self.index[i]])
            a_list.append(self.a[self.index[i]])
            s2_list.append(self.s2[self.index[i]])
            isterminal_list.append(self.isterminal[self.index[i]])
            r_list.append(self.r[self.index[i]])
            command_size_list.append(self.command_size[self.index[i]])
            command_size_next_list.append(self.command_size_next[self.index[i]])
            dead_e_index_list.append(self.dead_e_index[self.index[i]])
        return s1_list,a_list,s2_list,isterminal_list,r_list,command_size_list,command_size_next_list,dead_e_index_list
        # i = sample(range(0, self.size*2/3), sample_size)
    
        # return self.s1[i], self.a[i], self.s2[i], self.isterminal[i], self.r[i], self.command_size[i], self.command_size_next[i]

    def get_sample_validation(self):
        index = range(self.size*2/3, self.size)
        # print("sample_pos:{}".format(self.sample_pos))
        # print("self.size:{}".format(self.size))
        # self.sample_pos = min(self.sample_pos+sample_size,self.size)
        s1_list = []
        a_list = []
        s2_list = []
        isterminal_list = []
        r_list = []
        command_size_list = []
        command_size_next_list = []
        dead_e_index_list = []

        for i in index:
            s1_list.append(self.s1[self.index[i]])
            a_list.append(self.a[self.index[i]])
            s2_list.append(self.s2[self.index[i]])
            isterminal_list.append(self.isterminal[self.index[i]])
            r_list.append(self.r[self.index[i]])
            command_size_list.append(self.command_size[self.index[i]])
            command_size_next_list.append(self.command_size_next[self.index[i]])
            dead_e_index_list.append(self.dead_e_index[self.index[i]])

        return s1_list,a_list,s2_list,isterminal_list,r_list,command_size_list,command_size_next_list,dead_e_index_list
