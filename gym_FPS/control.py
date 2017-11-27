# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import signal
import numpy as np

class MyNPC(object):
    
    def __init__(self, env):
        self.env = env 
        self.ours = []
        
    def is_alive(self, plist):
#        print('test_alive',plist,self.ours)
        for bro in self.ours:
            if bro in plist:
                return bro
        return None
    
    def arrive(self, tar):
        
        players=self.env.getVal("players")
        searchplayer=self.env.getVal("poses")
        
        for bro in self.ours:
            if bro in players and self.small(searchplayer[bro], tar):
                print('ARRIVEDDDD',tar)
                return True
        return False
    
    def small(self, pos1, pos2, dis=10):
        distance=np.array(pos1)-np.array(pos2)
        gap=np.mat(distance)*np.mat(distance).T
        gap_double=np.sqrt(gap[0][0])
        if gap_double<dis:
            return True
        return False
    
class Scout(MyNPC):
    
    def __init__(self, me, env, mode, leader, targets, follows):
        super(self.__class__, self).__init__(env)
        self.mode = mode
        self.leader = leader
        self.ours.append(me)
        self.tars = targets
        self.follows = follows
    
    def wait_for_leader(self):
        print('chk',self.leader,self.env.getVal("wait_"+self.leader))
        if self.env.getVal("wait_"+self.leader) > 0:
            return True
        return False
    
    def run(self):
#        dead = False
        for i in range(len(self.tars)):
            tar = self.tars[i]
            self.env.move(tar, objid_list=self.ours,walkType='run')
        
        count = 0
        while count < 100:
            time.sleep(3)
            count+=1
            if self.wait_for_leader():
                print("get leader!")
                break
        
        last_pos = 0
        while True:
            pos = self.env.getVal("wait_"+self.leader)
            
            #we win
            if pos >= len(self.follows):
                break
            if last_pos != pos:
                self.env.can_attack_move(objid_list=self.ours,destPos=self.follows[pos])
                last_pos = pos
        
#        last_tar2 = None
#        while True:
#            tar2 = self.env.getVal(self.leader)
#            if last_tar2 != tar2:
#                self.env.move(tar2, objid_list=self.ours,walkType='run')
#            time.sleep(15)
        
class TeamWork(MyNPC):
    
    def __init__(self, label, env, ours, targets):
        super(self.__class__, self).__init__(env)
        self.tars = targets
        self.label = label
        for our in ours:
            self.ours.append(our)
            
    def wait_for_arrive(self, tar):
        count = 0
        last_pos = None
#        if tar[0]==-100.6:
#            time.sleep(15)
#            return False
        while True:
            time.sleep(5)
            print('test2',self.label)
            count+=1
            try:
                print('test3',self.label)
                playerlist=self.env.getVal("players")
                print('test4',self.label)
                searchplayer=self.env.getVal("poses")
                print('test5',self.label)
                bro = self.is_alive(playerlist)
                print('test6',self.label)
                if bro == None:
                    print('DDDDEADDDD!',self.label)
                    self.env.register('ready_'+self.label, True)
                    return True         
#                print('test7',self.label)
                pos = searchplayer[bro]
#                print('test8',self.label,pos)
    #            self.env.register(self.label, pos)
                
                if last_pos!=None and pos!=None and self.small(last_pos, pos):
                    count+=1
#                print('test9',self.label)
                if tar[0]!=-120.6:
                    if count >= 50 and tar[0]!=-58.6 or \
                        self.env.getVal("clear1") and tar[0]==-154.1:
                        print('CHANGE TO NEXT POSITION',self.label)
                        return False
                    else:
                        last_pos = pos
                
                print(self.label,'is arrived???',tar,pos,isinstance(pos, str))
                if isinstance(pos, str):
                    print('xxx1',self.label)
                    return False                 
                if self.arrive(tar):
                    return False
                
                try:
                    if tar[0]==-100.6 and not isinstance(pos, str):
                        if abs(tar[0]-pos[0])+abs(tar[2]-pos[2])<20:
                            print('!!!',abs(tar[0]-pos[0])+abs(tar[2]-pos[2]))
                            return False
                    if isinstance(pos, str):
                        print('xxx2',self.label)
                        return False
                except:
                    print('problem!')
                    pass
            except:
                print('kakaka!')
                if tar[0]==-120.6:
                    return False
                pass
        return False
    
    def run(self):
        for i in range(len(self.tars)):
            print('runit!',self.label,i)
            tar = self.tars[i]
            if i == len(self.tars)-1:
                print('final test',self.label,self.env.getVal('ready_team1'),self.env.getVal('ready_team2'))
                while not self.env.getVal('ready_team1') or \
                    not self.env.getVal('ready_team2'):
                        time.sleep(5)
            self.env.can_attack_move(objid_list=self.ours,destPos=tar)
            all_dead = self.wait_for_arrive(tar)
            if all_dead:
                print('all dead',self.label)
                break
            else:
                self.env.register("wait_"+self.label,i+1)
                print(self.label,'dbgx1',i,len(self.tars))
                if i == len(self.tars)-2: 
                    self.env.register('ready_'+self.label, True)

from threading import Thread
import functools

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

class Reader(object):
    def __init__(self, env):
        self.env = env
    
    def funct(self):
        try:
            players = self.env.get_objid_list()
            poses = self.env.get_objid_list(name=0,pos=1)
            self.env.register("players",players)
            self.env.register("poses",poses)
            
        except:
            print('read pass')
            pass
    
    def run(self):
        while True:
            func = timeout(timeout=7)(self.funct)
            try:
                func()
                time.sleep(3)
            except:
                print('timeout?')
                pass

class Moniter(object):
    
    def __init__(self, env):
        self.env = env
    
    def check(self, checks, code):
        if self.env.getVal(code)==True:
            return True
        
        playerdict=self.env.getVal("players")
        playerlist=[]
        for _,player_name in playerdict.items():
#            if player_name 
            playerlist.append(player_name)
        print('ck',checks)
        print('pl',playerlist)
        for ck in checks:
            if ck in playerlist:
                return False
        
        self.env.register(code,True)
        return True
    
    def run(self):
#        return None
        do_it = False
        while True:
            try:
                print("TASK:",self.env.getVal("clear1"),self.env.getVal("clear2"))
                clear1 = self.check(["驻守1","驻守2","驻守3","驻守4","驻守5"], "clear1")
                if clear1:
                    clear2 = self.check(["驻守8","驻守9","驻守10","驻守11"],"clear2")
                    if clear2:
                        print('DO IT!')
                        do_it = True
                        break
            except:
                print('TASK PASS')
                pass
            time.sleep(10)
            
#        self.env.search_enemy_attack()