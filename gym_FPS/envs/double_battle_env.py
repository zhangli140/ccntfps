# coding=utf-8
from __future__ import division
import numpy as np
import math
from gym import spaces
from .. import utils  

from . import FPS_env as fc
import time
from .starcraft.Config import *
from queue import deque
import copy
import threading
import sys, random

DISTANCE_FACTOR = 16
ENEMY = 1
MYSELF = 0
#point_list=[[125,-1,185],[205,-1,105],[125,-1,25],[45,-1,105],[125,-1,105],[125,-1,145],[165,-1,105],[125,-1,65],[85,-1,105]]
point_list=[[125,-1,175],[205,-1,95],[125,-1,15],[45,-1,95],[125,-1,95],[125,-1,135],[165,-1,95],[125,-1,55],[85,-1,95]]
class doubleBattleEnv(fc.FPSEnv):
    def __init__(self):
        super(doubleBattleEnv, self).__init__()

        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        # self.state['myunits'] = {}
        self.myunits = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.units_id = []
        self.units_e_id = []
        self.units_dead_id = []
        self.current_my_units = {}
        self.current_enemy_units = {}
        self.episodes = 0
        self.episode_steps = 0
        self.init_my_units = {}
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.flag = True
        self.time1 = 0
        self.time2 = 0
        self.stop = False
        self.sup_outside = [[],[],[],[]]
        self.sup_inside = [[],[],[],[],[]]
        self.pre_obs_myself = np.zeros((9,2), dtype=int)
        self.pre_obs_enemy = np.zeros((9,2), dtype=int)

        self.pushed_cmd_excuting = dict()
        self.pushed_cmd_excuting_switch = True

        self.init_coords = None


    def _action_space(self):
        action_low = [-1.0, -math.pi/2, -1.0]
        action_high = [1.0, math.pi/2, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        obs_low = np.zeros([1, 10])
        obs_high = (np.zeros([1, 10]) + 1) * 100
        return spaces.Box(np.array(obs_low), np.array(obs_high))

    def reset_fight(self):

        self.new_epidode_flag = True

        self._make_feature()
        self.screen_my, self.screen_enemy = self._make_observation()
        self.obs = dict()
        self.obs['screen_my'] = self.screen_my
        self.obs['screen_enemy'] = self.screen_enemy
        self.init_my_units = self.state['units_myself']

        self.pushed_cmd_excuting = dict()

        th2 = threading.Thread(target=self.pushedCmdMonitor)
        th2.start()

        self.mc = list()
        if 'client_enemy' in self.units.values():
            self.mc.append(utils.get_key_from_value(d=self.units, v='client_enemy'))
        
        return self.obs

    def _reset(self):

        self.episodes += 1
        self.episode_steps = 0
        self.flag = 0
        if not self.is_enemy:
            self.new_episode(disablelog=0)
        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        # self.state['myunits'] = {}
        self.myunits = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.units_id = []
        self.units_e_id = []
        self.units_dead_id = []
        self.current_my_units = {}
        self.current_enemy_units = {}
        self.new_epidode_flag = True
        self.init_my_units = {}
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.time1 = 0
        self.time2 = 0
        self.state['game_over'] = False
        self.state['win'] = False
        self.pushed_cmd_excuting = dict()
        #print("while1")
        time.sleep(10)

        if not self.is_enemy:
            time.sleep(10)
            if not self.is_5player:
                self.add_obj(name="敌人1", is_enemy=True, pos=[125, -1, 100], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人2", is_enemy=True, pos=[125, -1, 101], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人3", is_enemy=True, pos=[125, -1, 99], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人4", is_enemy=True, pos=[124, -1, 100], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人5", is_enemy=True, pos=[126, -1, 100], leader_objid=-1, team_id=-1)

                self.add_obj(name="敌人6", is_enemy=True, pos=[124, -1, 101], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人7", is_enemy=True, pos=[126, -1, 101], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人8", is_enemy=True, pos=[124, -1, 99], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人9", is_enemy=True, pos=[126, -1, 99], leader_objid=-1, team_id=-1)
                self.add_obj(name="敌人10", is_enemy=True, pos=[127, -1, 100], leader_objid=-1, team_id=-1)

            self.add_obj(name="队友1", is_enemy=False, pos=[125, -1, 180], leader_objid=-1, team_id=1)
            self.add_obj(name="队友2", is_enemy=False, pos=[125, -1, 181], leader_objid=-1, team_id=1)
            self.add_obj(name="队友3", is_enemy=False, pos=[125, -1, 179], leader_objid=-1, team_id=1)
            self.add_obj(name="队友4", is_enemy=False, pos=[124, -1, 180], leader_objid=-1, team_id=1)
            self.add_obj(name="队友5", is_enemy=False, pos=[126, -1, 180], leader_objid=-1, team_id=1)

            self.add_obj(name="队友6", is_enemy=False, pos=[124, -1, 181], leader_objid=-1, team_id=1)
            self.add_obj(name="队友7", is_enemy=False, pos=[126, -1, 181], leader_objid=-1, team_id=1)
            self.add_obj(name="队友8", is_enemy=False, pos=[124, -1, 179], leader_objid=-1, team_id=1)
            self.add_obj(name="队友9", is_enemy=False, pos=[126, -1, 179], leader_objid=-1, team_id=1)
            self.add_obj(name="队友10", is_enemy=False, pos=[127, -1, 180], leader_objid=-1, team_id=1)
            #print('finish add ai')
            time.sleep(Config.sleeptime)
        self.myunits={}
        self._make_feature()
        #self.screen_my, self.screen_enemy = self._make_observation()
        #self.obs = dict()
        #self.obs['screen_my'] = self.screen_my
        #self.obs['screen_enemy'] = self.screen_enemy
        self.init_my_units = self.state['units_myself']
        #unit_size = len(self.state['units_myself'])
        #unit_size_e = len(self.state['units_enemy'])
        #self.add_observer([220, -1, 20], 2000) # disable the fog of war
        self.pre_obs_myself = np.zeros((9,2), dtype=float)
        self.pre_obs_enemy = np.zeros((9,2), dtype=float)
        self.sup_outside = [[],[],[],[]]
        self.sup_inside = [[],[],[],[],[]]
        f_obs1, f_obs2 = self.decay_feature()

        return f_obs1, f_obs2

    def _wait_for_open_panel(self,):
        '''
        决定当面板打开时应该显示什么
        '''     
        def get_word(l):
            '''
            根据分兵策略生成战术面板文字
            '''
            res = []
            words = ['北门 ', '东门 ', '南门 ', '西门 ', '驻守中心 ']
            for i, num in enumerate(l):
                if num > 0:
                    res.append(words[i] + str(int(num)))
            return ','.join(res)

        def get_point(l):
            '''
            根据分兵策略生成战术面板提示绿点
            '''
            if self.is_enemy:
                point_list = [ [125, -1, 95], [125, -1, 135],[165, -1, 95], [125, -1, 55], [85, -1, 95]]
            else:
                point_list = [[125, -1, 175], [205, -1, 95], [125, -1, 15], [45, -1, 95]]
            res = []
            for i, num in enumerate(l):
                for _ in range(int(num)):
                    res.append(point_list[i])

            for i in range(len(res)):
                res[i][0] += (i - 5) // 3
                res[i][2] += (i - 5) % 3

            return res


        def outer():
            '''
            by xiaoxiang
            '''
            def getpos(curpos):
                p1=[125,175]
                p2=[205,95]
                p3=[125,15]
                p4=[45,95]
                p5=[125,95]
                p6=[125,125]
                p7=[155,95]
                p8=[125,65]
                p9=[95,95]
                outerradius=30
                innerradius=15
                matrixcenter=[p1,p2,p3,p4,p5,p6,p7,p8,p9]
                if isinner(curpos,matrixcenter[0],outerradius):
                    return 1
                if isinner(curpos,matrixcenter[1],outerradius):
                    return 2
                if isinner(curpos,matrixcenter[2],outerradius):
                    return 3
                if isinner(curpos,matrixcenter[3],outerradius):
                    return 4
                if isinner(curpos,matrixcenter[4],innerradius):
                    return 5
                if isinner(curpos,matrixcenter[5],innerradius):
                    return 6
                if isinner(curpos,matrixcenter[6],innerradius):
                    return 7
                if isinner(curpos,matrixcenter[7],innerradius):
                    return 8
                if isinner(curpos,matrixcenter[8],innerradius):
                    return 9
                return 0

            def isinner(curpos,area,radius):
                leftbottom=[area[0]-radius,area[1]-radius]
                righttop=[area[0]+radius,area[1]+radius]
                flag=False
                if leftbottom[0]<=curpos[0] and leftbottom[1]<=curpos[1] and righttop[0]>=curpos[0] and righttop[1]>=curpos[1]:
                    flag=True
                return flag

            def getfeature():
                '''
                changed some to speedup
                '''
                curfeaturepos=self.states
                curenemylist=[0,0,0,0,0]
                self._make_feature()
                enemyteam=self.state['units_enemy']
                #comradeteam=self.state['units_myself']
                for playerid,state in curfeaturepos.items():
                    if playerid in enemyteam:
                        temp=state["POSITION"]
                        curpos=[temp[0],temp[2]]
                        index=getpos(curpos)
                        if index>0:
                            curenemylist[index-5]+=1
                            
                data, result=[], []
                for i in range(11):
                    for j in range(11-i):
                        for k in range(11-i-j):
                            m=10-i-j-k
                            result.append([i,j,k,m])
                            curposfeature=[i,j,k,m]
                            curposfeature.extend(curenemylist)
                            data.append(curposfeature)
                            
                pre = self.clf.predict_proba(data)[:,1]
                return result[np.argmax(pre)]
            return getfeature()


        while True:
            s = self.client.strategy_open
            if len(s) > 0:
                print('receive open')
                d = {'Items': []}
                str1 = get_word(self.assignment[0])
                str2 = get_word(self.assignment[1])
                green_point1 = get_point(self.assignment[0])
                green_point2 = get_point(self.assignment[1])
                if not self.is_enemy:
                    print(outer())
                    res_xx = outer()
                    self.assignment[2] = res_xx
                    str3 = get_word(res_xx)
                    green_point3 = get_point(res_xx)
                    self.add_strategy('分兵', [str1, str2, str3], [green_point1, green_point2, green_point3], d)
                    self.add_strategy('进攻', ['同时进入'], [[[125, -1, 100]]], d)
                else:
                    self.add_strategy('分兵', [str1, str2], [green_point1, green_point2], d)
                    c_pos_list = []
                    for _, ut in self.states.items():
                        if (ut['TEAM_ID'] < 0 and self.is_enemy) or (ut['TEAM_ID'] > 0 and not self.is_enemy):
                            c_pos_list.append(ut['POSITION'])
                    self.add_strategy('驻守', ['原地防御'], [c_pos_list], d)
                self.open_strategy_panel(d)

                self.client.strategy_open=''


    def _voice_check(self,):
        '''
        检查并处理语音指令
        '''
        pos_list = [[], 
            [125, -1, 95], [125, -1, 135], [165, -1, 95], [125, -1, 55], [85, -1, 95]
        ]
        def work(s, word):            
            try:
                l = s.split(word)
                team_id = l[0]
                if l[1][-1] == '号':
                    target_id = int(l[1][:-1])
                    if self.is_enemy: #skip bug that server idx != client idx
                        target_id -= 1
                    flag = 0
                else:
                    flag = 1
                    if l[1][-1] == '区':
                        area_id = int(l[1][:-1])
                    else:
                        area_id = int(l[1])
                if team_id.find('队') > -1:
                    if self.is_enemy:
                        objid_list = self.enemy_team_member[-int(team_id[:-1])]
                    else:
                        objid_list = self.team_member[int(team_id[:-1])]
                elif team_id[-1] == '号':
                    objid_list = [int(team_id[:-1])]
                    if self.is_enemy: #skip bug that server idx != client idx
                        objid_list[0] -= 1
                else:
                    objid_list = [int(team_id)]
                for uid in objid_list:
                    if flag == 1:
                        pce = [1, pos_list[area_id][0], pos_list[area_id][2]]
                    else:
                        pce = [0, target_id]
                    self.pushed_cmd_excuting[uid] = pce
                self.origin_ai(objid_list=objid_list, move_attack=False)
                if flag == 1:
                    self.move(destPos=pos_list[area_id], objid_list=objid_list, walkType='run', pos='head')
                else:
                    self.setTargetAndAttack(objid_list=objid_list, targetObjID=target_id)
                print(objid_list)
            except:
                self.add_chat('解析失败', 0, -1)

        def plan(s):
            idx = int(s[2:])
            team_num = [[0,2,2,2,2,2], [0,2,5,1,1,1], [0,0,3,3,2,2], [0,0,1,2,5,2]][idx]
            area_id = 1
            objid_list = []
            while team_num[area_id] == 0:
                area_id += 1
            for uid, u_data in self.states.items():
                if u_data['TEAM_ID'] < 0 and self.units[uid] != 'client_enemy':
                    team_num[area_id] -= 1
                    objid_list.append(uid)
                    while team_num[area_id] == 0:
                        print('run team %d'%area_id)
                        self.origin_ai(objid_list=objid_list, move_attack=False)
                        self.move(destPos=pos_list[area_id], objid_list=objid_list, walkType='run', pos='head')
                        area_id += 1
                        objid_list = []
                        if area_id > 5:
                            break


        def analyse(s):
            try:
                if s=='重算':
                    self.stop=True
                    self.pause=False
                elif s=='暂停':
                    self.stop=True
                    self.pause=True                    
                elif s.find('方案') > -1:
                    if self.is_enemy:
                        plan(s)
                elif s.find('撤退') > -1:
                    work(s, '撤退')
                elif s.find('转进') > -1:
                    work(s, '转进')
                elif s.find('攻击') > -1:
                    work(s, '攻击')
                elif s.find('进攻') > -1:
                    work(s, '进攻')
                elif s.find('支援') > -1:
                    work(s, '支援')
                elif s.find('编队') > -1:
                    pass
            except:
                print('except')
            
        while True:
            if len(self.server_voice.buff) > 0:
                s = self.server_voice.buff
                print('voice_check:', s)
                if self.is_enemy:
                    self.enemy_client.confirm = ''
                else:
                    self.client.confirm = ''
                self.server_voice.buff = ''
                self.add_chat(s, 0, -1)
                for _ in range(20):
                    if self.is_enemy:
                        if len(self.enemy_client.confirm) > 0:
                            self.add_chat('语音指令确认' + s, 0, -1)
                            analyse(s)
                            self.enemy_client.confirm = ''
                            break
                    else:
                        if len(self.client.confirm) > 0:
                            self.add_chat('语音指令确认' + s, 0, -1)
                            analyse(s)
                            self.client.confirm = ''
                            break
                    time.sleep(0.1)

    def pushedCmdMonitor(self):
        while self.pushed_cmd_excuting_switch and not self.state['game_over']:
            pushed_cmd_excuting_ = dict()
            d = self.pushed_cmd_excuting.copy()
            for uid, cmd in d.items():
                if cmd[0] == 0:
                    # attack
                    uid_e = cmd[1]
                    if uid in self.states.keys() and uid_e in self.states.keys() and self.states[uid]['HEALTH'] and self.states[uid_e]['HEALTH']:
                        pushed_cmd_excuting_[uid] = cmd
                elif cmd[0] == 1:
                    # move
                    if not uid in self.states.keys():
                        continue
                    x1, y1 = self.states[uid]['POSITION'][0], self.states[uid]['POSITION'][2]
                    x2, y2 = cmd[1], cmd[2]
                    if self.states[uid]['HEALTH'] and utils.get_distance(x1, y1, x2, y2) >= 10:
                        pushed_cmd_excuting_[uid] = cmd
            self.pushed_cmd_excuting = pushed_cmd_excuting_.copy()

    def _action2cmd(self,action,uid,flag):
        ut = self.myunits[uid]
        x = ut['POSITION'][0]
        y = ut['POSITION'][2]
        # print("before action: (x,y):({},{})".format(x,y))
        dist = 30
        if action == 0:
            y -= dist
        elif action == 1:
            y += dist
        elif action == 2:
            x -= dist
        elif action == 3:
            x += dist
        elif action == 4:
            x += dist
            y -= dist
        elif action == 5:
            x += dist
            y += dist
        elif action == 6:
            x -= dist
            y += dist
        elif action == 7:
            x -= dist
            y -= dist
        elif action == 8:
            pass
        else:
            ut = {}
            if flag is 'myself':
                e_uid = self.units_e_id[action-9]
                ut = self.myunits[e_uid]
            else:
                e_uid = self.units_id[action-9]
                ut = self.myunits[e_uid]
            x = ut['POSITION'][0]
            y = ut['POSITION'][2]
        return x,y

        
    def _make_commands(self, action, flag):
        cmds = []
        self.current_my_units = self.state['units_myself']
        self.current_enemy_units = self.state['units_enemy']
        if self.state is None or (len(action) == 0):
            return cmds
        if flag == 'myself':
            if len(action) is not len(self.units_id):
                return cmds
            # for uid, ut in self.state['units_myself'].items():
            for i in range(len(self.units_id)):
                uid = self.units_id[i]
                ut = self.myunits[uid]
                myself = ut
                if ut['HEALTH']<=0:
                    continue
                if action[i] > 8:
                    # Attack action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    enemy_id, distance = utils.get_closest_except(x2, y2, self.state['units_enemy'], exp=self.mc)
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2, y2 = self._action2cmd(action[i], uid, flag)
                    enemy_id, distance = utils.get_closest_except(x2, y2, self.state['units_enemy'], exp=self.mc)
                    cmds.append([0, uid, enemy_id])

        else:
            # only has attack command
            if len(action) is not len(self.units_e_id):
                return cmds

            for i in range(len(self.units_e_id)):
                uid = self.units_e_id[i]
                ut = self.myunits[uid]
                myself = ut
                if ut['HEALTH']<=0:
                    continue
                if action[i] > 8:
                    # Attack action
                    if myself is None:
                        return cmds
                    x2,y2 = self._action2cmd(action[i], uid,flag)
                    enemy_id, distance = utils.get_closest_except(x2, y2, self.state['units_myself'], exp=self.mc)
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2, y2 = self._action2cmd(action[i], uid, flag)
                    enemy_id, distance = utils.get_closest_except(x2, y2, self.state['units_myself'], exp=self.mc)
                    cmds.append([0, uid, enemy_id])


        # print "commands send!"
        return cmds

    def die_fast(self):
       # count_them = len(self.state['units_enemy'])
        cx_e, cy_e = utils.get_units_center(self.state['units_enemy'])
        cx, cy = utils.get_units_center(self.state['units_myself'])

        for uid, _ in self.state['units_myself'].items():
            self.move(objid_list=[uid], destPos=[cx_e, -1, cy_e], reachDist=3, walkType='run')
       # count_us = len(self.state['units_myself'])
        for uid, _ in self.state['units_enemy'].items():
            self.move(objid_list=[uid], destPos=[cx, -1, cy], reachDist=3, walkType='run')

        time.sleep(Config.sleeptime)
        self._make_feature()
        done = self.state['game_over']
        return done

    def _step(self, actions):
        print('double step')
        self.episode_steps += 1
        action = actions[0].copy()
        action_e = actions[1].copy()
        
        commands = self._make_commands(action, 'myself')
        commands_e = self._make_commands(action_e, 'enemy')
        print('commands', commands)
        print('commands_e', commands_e)

        self.current_my_units = copy.deepcopy(self.state['units_myself'])
        self.current_enemy_units = copy.deepcopy(self.state['units_enemy'])
        cmdThreads = []
        for i in range(len(commands)):
            if commands[i][1] in self.pushed_cmd_excuting.keys():
                continue
            if commands[i][0] == 0:
                if commands[i][2] == -1:
                    continue
                unit = self.states[commands[i][2]]
                self.states[commands[i][1]]['LAST_CMD']=[0, unit['POSITION'][0], unit['POSITION'][2]]
                self.set_target_objid(objid_list=[commands[i][1]], targetObjID=commands[i][2])
                # self.attack(objid_list=[commands[i][1]], auth='normal', pos='replace')
                cmdThreads.append(threading.Thread(target=self.attack,
                                                   kwargs={'objid_list': [commands[i][1]], 'auth': 'normal',
                                                           'pos': 'replace'}))
            else:
                self.states[commands[i][1]]['LAST_CMD'] = [1, commands[i][2][0], commands[i][2][2]]
                # self.move(objid_list=[commands[i][1]], destPos=commands[i][2], reachDist=3, walkType='run')
                cmdThreads.append(threading.Thread(target=self.moveAndAttackClosest,
                                                   kwargs={'objid_list': [commands[i][1]], 'destPos': commands[i][2],
                                                           'reachDist': 3, 'walkType': 'run',
                                                           'enemy': self.state['units_enemy']}))
            self.states[commands[i][1]]['LAST_TIME'] = self.states[commands[i][1]]['TIME']
            self.states[commands[i][1]]['TIME'] = time.time()
            self.states[commands[i][1]]['LAST_POSITION_'] = self.states[commands[i][1]]['POSITION']
        # print('myself commands sent')
        for i in range(len(commands_e)):
            if commands_e[i][1] in self.pushed_cmd_excuting.keys():
                continue
            if commands_e[i][0] == 0:
                if commands_e[i][2] == -1:
                    continue
                unit = self.states[commands_e[i][2]]
                self.states[commands_e[i][1]]['LAST_CMD'] = [0, unit['POSITION'][0], unit['POSITION'][2]]
                self.set_target_objid(objid_list=[commands_e[i][1]], targetObjID=commands_e[i][2])
                # self.attack(objid_list=[commands_e[i][1]], auth='normal', pos='replace')
                cmdThreads.append(threading.Thread(target=self.attack,
                                                   kwargs={'objid_list': [commands_e[i][1]], 'auth': 'normal',
                                                           'pos': 'replace'}))
            else:
                self.states[commands_e[i][1]]['LAST_CMD'] = [1, commands_e[i][2][0], commands_e[i][2][2]]
                # self.move(objid_list = [commands_e[i][1]], destPos=commands_e[i][2], reachDist=3, walkType='run')
                cmdThreads.append(threading.Thread(target=self.moveAndAttackClosest,
                                                   kwargs={'objid_list': [commands_e[i][1]], 'destPos': commands_e[i][2],
                                                           'reachDist': 3, 'walkType': 'run',
                                                           'enemy': self.state['units_myself']}))
            self.states[commands_e[i][1]]['LAST_TIME'] = self.states[commands_e[i][1]]['TIME']
            self.states[commands_e[i][1]]['TIME'] = time.time()
            self.states[commands_e[i][1]]['LAST_POSITION_'] = self.states[commands_e[i][1]]['POSITION']

        random.shuffle(cmdThreads)
        self.cmdThreadsLen = len(cmdThreads)
        for th in cmdThreads:
            th.setDaemon(True)
            th.start()
        while self.cmdThreadsLen > 0:
            time.sleep(0.1)
        print('commands sent')

        time.sleep(Config.sleeptime)
        self._make_feature()
        self.obs = dict()
        self.screen_my, self.screen_enemy = self._make_observation()
        self.obs['screen_my'] = self.screen_my
        self.obs['screen_enemy'] = self.screen_enemy
        reward = self._compute_reward()
        print('reward', reward)
        done = self.state['game_over']
        unit_size = len(self.units_id)
        unit_size_e = len(self.units_e_id)
        return self.obs, reward, done, unit_size, unit_size_e

    def _make_feature(self):
        # init
        if len(self.myunits) == 0:
            for uid, ut in self.states.items():

                if ut['TEAM_ID'] > 0 and uid != 0 and ut['HEALTH'] > 0:
                    self.state['units_myself'][uid] = self.states[uid]
                    self.myunits[uid] = self.states[uid]
                    self.units_id.append(uid)

                elif ut['TEAM_ID'] <= 0 and uid != 0 and ut['HEALTH'] > 0:

                    self.state['units_enemy'][uid] = self.states[uid]
                    self.units_e_id.append(uid)
                    self.myunits[uid] = self.states[uid]
                else:
                    print("ut:{}".format(ut))

                    
            # print("for end!")
            self.units_id.sort()
            self.units_e_id.sort()

        # update
        else:
            pass
        
        for uid, ut in self.states.items():
            if ut['TEAM_ID'] > 0:
                if uid != 0 and ut['HEALTH'] > 0:
                    self.myunits[uid] = self.states[uid]
                    # print("self.state['units_myself'][uid] :{}".format(self.state['units_myself'][uid] ))
                elif uid!=0:
                    if uid not in self.units_dead_id:
                        self.units_dead_id.append(uid)
            else:
                if uid != 0 and ut['HEALTH'] > 0:
                    self.myunits[uid] = self.states[uid]
                elif uid!=0:
                    if uid not in self.units_dead_id:
                        self.units_dead_id.append(uid)
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        for uid, ut in self.states.items():
            if ut['TEAM_ID'] > 0 and uid != 0 and ut['HEALTH'] > 0:
                self.state['units_myself'][uid] = ut              # alive
        
            elif uid != 0 and ut['HEALTH'] > 0:
                self.state['units_enemy'][uid] = ut           # alive
        if len(self.state['units_myself']) == 0 or len(self.state['units_enemy']) == 0:
            self.state['game_over'] = True
            if len(self.state['units_myself']) > 0:
                self.state['battle_won'] = True
            else:
                self.state['battle_won'] = False
        else:
            self.state['game_over'] = False

    def _make_observation(self):
        resolution = (255, 255)

        # ---------------------myself--------------------------------
        screen_my_list = []
        for me_ind in range(len(self.units_id)):
            screen_my = np.zeros(resolution, dtype=int)
            uid_me = self.units_id[me_ind]
            ut_me = self.myunits[uid_me]
            for ind in range(len(self.units_id)):
                uid = self.units_id[ind]
                ut = self.myunits[uid]
                if ut['HEALTH'] > 0:
                    center_y = (int(round(ut['POSITION'][0])) - int(round(ut_me['POSITION'][0]))) // 2 + resolution[1] // 2
                    center_x = (int(round(ut['POSITION'][2])) - int(round(ut_me['POSITION'][2]))) // 2 + resolution[0] // 2
                    if uid == uid_me:
                        screen_my[center_x][center_y] = 20
                    else:
                        screen_my[center_x][center_y] = 2
            for ind in range(len(self.units_e_id)):
                uid = self.units_e_id[ind]
                ut = self.myunits[uid]
                if ut['HEALTH'] > 0:
                    center_y = (int(round(ut['POSITION'][0])) - int(round(ut_me['POSITION'][0]))) // 2 + resolution[1] // 2
                    center_x = (int(round(ut['POSITION'][2])) - int(round(ut_me['POSITION'][2]))) // 2 + resolution[0] // 2
                    screen_my[center_x][center_y] = 4

            screen_my_list.append(screen_my)

        # ---------------------enemy--------------------------------
        screen_enemy_list = []
        for me_ind in range(len(self.units_e_id)):
            screen_enemy = np.zeros(resolution, dtype=int)
            uid_me = self.units_e_id[me_ind]
            ut_me = self.myunits[uid_me]
            for ind in range(len(self.units_id)):
                uid = self.units_id[ind]
                ut = self.myunits[uid]
                if ut['HEALTH'] > 0:
                    center_y = (int(round(ut['POSITION'][0])) - int(round(ut_me['POSITION'][0]))) // 2 + resolution[1] // 2
                    center_x = (int(round(ut['POSITION'][2])) - int(round(ut_me['POSITION'][2]))) // 2 + resolution[0] // 2
                    screen_enemy[center_x][center_y] = 4

            for ind in range(len(self.units_e_id)):
                uid = self.units_e_id[ind]
                ut = self.myunits[uid]
                if ut['HEALTH'] > 0:
                    center_y = (int(round(ut['POSITION'][0])) - int(round(ut_me['POSITION'][0]))) // 2 + resolution[1] // 2
                    center_x = (int(round(ut['POSITION'][2])) - int(round(ut_me['POSITION'][2]))) // 2 + resolution[0] // 2
                    if uid == uid_me:
                        screen_enemy[center_x][center_y] = 20
                    else:
                        screen_enemy[center_x][center_y] = 2

            screen_enemy_list.append(screen_enemy)

        return screen_my_list, screen_enemy_list

    def _compute_reward(self):

        if len(self.current_my_units) == 0 or len(self.current_enemy_units) == 0:
            return None

        last_my = 0
        last_enemy = 0
        new_my = 0
        new_enemy = 0

        for uid, ut in self.current_my_units.items():      #action执行前
            last_my += ut['HEALTH'] + 50

        for uid, ut in self.current_enemy_units.items():
            last_enemy += ut['HEALTH'] + 50

        for uid, ut in self.state['units_myself'].items():
            new_my += ut['HEALTH'] + 50

        for uid, ut in self.state['units_enemy'].items():
            new_enemy += ut['HEALTH'] + 50

        return (new_my - new_enemy) - (last_my - last_enemy)

    def formation_reward(self):
        tmp = 0
        for uid, ut in self.state['units_myself'].items():
            tmp += ut['HEALTH']

        for uid, ut in self.state['units_enemy'].items():
            tmp -= ut['HEALTH']

        return tmp

    def get_formation_feature(self):
        self._make_feature()

        f_obs1 = np.zeros((9,2),dtype=float)
        f_obs2 = np.zeros((9,2),dtype=float)

        pos = point_list
        for uid,ut in self.state['units_myself'].items():
            for i in range(4):
                if pos[i][0]-10<=ut['POSITION'][0] and pos[i][0]+10>=ut['POSITION'][0] and pos[i][2]-10<=ut['POSITION'][2] and pos[i][2]+10>=ut['POSITION'][2]:
                    f_obs1[i][0] += 1
                    f_obs1[i][1] += ut['HEALTH']
                    continue
                #elif uid in self.sup_outside[i]:
                    #f_obs1[i][2] += 1
                    #f_obs1[i][3] += ut['HEALTH']
                    continue
        for uid,ut in self.state['units_enemy'].items():
            for i in range(4,9):
                if pos[i][0]-10<=ut['POSITION'][0] and pos[i][0]+10>=ut['POSITION'][0] and pos[i][2]-10<=ut['POSITION'][2] and pos[i][2]+10>=ut['POSITION'][2]:
                    f_obs2[i][0] += 1
                    f_obs2[i][1] += ut['HEALTH']
                    continue
                #elif uid in self.sup_inside[i-4]:
                    #f_obs2[i][2] += 1
                    #f_obs2[i][3] += ut['HEALTH']
                    #continue
        for i in range(9):
            if f_obs1[i][0]>0:
                f_obs1[i][1] /= f_obs1[i][0]
            #if f_obs1[i][2]>0:
                #f_obs1[i][3] /= f_obs1[i][2]
            if f_obs2[i][0]>0:
                f_obs2[i][1] /= f_obs2[i][0]
            #if f_obs2[i][2]>0:
                #f_obs2[i][3] /= f_obs2[i][2]

        return f_obs1, f_obs2

    def decay_feature(self):
        sub_obs1,sub_obs2 = self.get_formation_feature()
        sub_obs1 = np.array(sub_obs1)
        sub_obs2 = np.array(sub_obs2)
        f_obs1 = np.hstack((sub_obs1,self.pre_obs_enemy))
        f_obs2 = np.hstack((sub_obs2,self.pre_obs_myself))
        self.pre_obs_myself = sub_obs1[:, 0:2]
        self.pre_obs_enemy = sub_obs2[:, 0:2]
        f_obs2 = np.vstack((f_obs2[4:9, :], f_obs2[0:4, :]))
        return f_obs1, f_obs2

                


