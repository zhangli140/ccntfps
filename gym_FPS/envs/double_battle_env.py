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
        self.add_observer([220, -1, 20], 2000) # disable the fog of war
        self.pre_obs_myself = np.zeros((9,2), dtype=float)
        self.pre_obs_enemy = np.zeros((9,2), dtype=float)
        self.sup_outside = [[],[],[],[]]
        self.sup_inside = [[],[],[],[],[]]
        f_obs1, f_obs2 = self.decay_feature()

        return f_obs1, f_obs2

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
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_enemy'])
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2, y2 = self._action2cmd(action[i], uid, flag)
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_enemy'])
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
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_myself'])
                    cmds.append([0, uid, enemy_id])
                elif action[i] is not -1:
                    # Move action
                    if myself is None:
                        return cmds
                    x2, y2 = self._action2cmd(action[i], uid, flag)
                    enemy_id, distance = utils.get_closest(x2, y2, self.state['units_myself'])
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

            # if self.new_epidode_flag:
            #     self.new_epidode_flag = False
            #     cmdThreads.append(threading.Thread(target=self.origin_ai,
            #                                        kwargs={'objid_list': self.state['units_enemy'].keys()}))

            # for ind in range(len(self.units_e_id)):
            #     uid = self.units_e_id[ind]
            #     ut = self.myunits[uid]
            #     if ut['HEALTH'] > 0:
            #         x, y = ut['POSITION'][0], ut['POSITION'][2]
            #         enemy_id, distance = utils.get_closest(x, y, self.state['units_myself'])
            #         self.set_target_objid(objid_list=[uid], targetObjID=enemy_id)
            #         cmdThreads.append(threading.Thread(target=self.attack,
            #                                            kwargs={'objid_list': [uid], 'auth': 'normal',
            #                                                    'pos': 'replace'}))

            # # 注意：以下注释以敌方视角叙述
            # # 敌我士兵按照距离外部四点的距离归类到四个战场
            # pointsSolsMyself, pointsSolsEnemy = self.gatherPointsSoldiers()
            # # 计算在每个战场敌我士兵的总血量（除去正在支援的士兵）
            # pointsHealthSumMyself, pointsHealthSumEnemy = [0 for _ in range(len(self.points_out))], [0 for _ in range(
            #     len(self.points_out))]
            # for i in range(len(self.points_out)):
            #     for _, uid in enumerate(pointsSolsEnemy[i]):
            #         if uid in self.support_list.keys():
            #             continue
            #         ut = self.myunits[uid]
            #         pointsHealthSumEnemy[i] += ut['HEALTH']
            #     for _, uid in enumerate(pointsSolsMyself[i]):
            #         ut = self.myunits[uid]
            #         pointsHealthSumMyself[i] += ut['HEALTH']
            # # 若支援士兵已到达支援战场，或者要支援的战场里队友或者敌人都死光了，就把此人从支援列表里移除
            # support_list_ = self.support_list.copy()
            # for uid, pos in self.support_list.items():
            #     for i in range(len(self.points_out)):
            #         if uid in pointsSolsEnemy[i]:
            #             ut = self.myunits[uid]
            #             if pos == i or pointsHealthSumEnemy[pos] == 0 or pointsHealthSumMyself[pos] == 0:
            #                 support_list_.pop(uid)
            #                 pointsHealthSumEnemy[i] += ut['HEALTH']
            # self.support_list = support_list_
            # # 对每个战场计算一个支援权重，若在某个战场敌方总血量小于我方，则权重为0，反之，设置权重为（敌方总血量/我方总血量）
            # pointsSupportProp = []
            # for i in range(len(self.points_out)):
            #     if pointsHealthSumEnemy[i] != 0 and pointsHealthSumEnemy[i] < pointsHealthSumMyself[i]:
            #         pointsSupportProp.append(pointsHealthSumMyself[i] / pointsHealthSumEnemy[i])
            #     else:
            #         pointsSupportProp.append(0)
            # # 归一化
            # if sum(pointsSupportProp) != 0:
            #     pointsSupportProp = [it / sum(pointsSupportProp) for it in pointsSupportProp]
            #
            # # 指令阶段，遍历第i个战场
            # for i in range(len(self.points_out)):
            #     # 若第i个战场我方总血量低于或等于敌方，则不支援，此战场的士兵攻击距离自己最近的目标
            #     if pointsHealthSumEnemy[i] <= pointsHealthSumMyself[i]:
            #         for _, uid in enumerate(pointsSolsEnemy[i]):
            #             # 支援列表中的士兵（说明还没有支援到位）不做处理，以免给其下达新的指令打断支援过程
            #             if uid in self.support_list.keys():
            #                 continue
            #             ut = self.myunits[uid]
            #             x, y = ut['POSITION'][0], ut['POSITION'][2]
            #             enemy_id, distance = utils.get_closest(x, y, self.state['units_myself'])
            #             self.set_target_objid(objid_list=[uid], targetObjID=enemy_id)
            #             cmdThreads.append(threading.Thread(target=self.attack,
            #                                                kwargs={'objid_list': [uid], 'auth': 'normal',
            #                                                        'pos': 'replace'}))
            #     # 否则，按策略抽调一批增援部队
            #     else:
            #         # 增援队列，只有在队列内的士兵才被允许前往支援（是否支援，支援哪里要根据下面策略决策）
            #         supporter = []
            #         pointHealthSum = pointsHealthSumEnemy[i]
            #         for _, uid in enumerate(pointsSolsEnemy[i]):
            #             if uid in self.support_list.keys():
            #                 continue
            #             ut = self.myunits[uid]
            #             # 加入支援队列的士兵总血量不得超过在此据点的敌我血量差，防止为了增援而把优势打成劣势
            #             if pointHealthSum - ut['HEALTH'] >= pointsHealthSumMyself[i]:
            #                 supporter.append(uid)
            #                 pointHealthSum -= ut['HEALTH']
            #             # 余下的士兵行为与不支援的据点士兵一样
            #             else:
            #                 x, y = ut['POSITION'][0], ut['POSITION'][2]
            #                 enemy_id, distance = utils.get_closest(x, y, self.state['units_myself'])
            #                 self.set_target_objid(objid_list=[uid], targetObjID=enemy_id)
            #                 cmdThreads.append(threading.Thread(target=self.attack,
            #                                                    kwargs={'objid_list': [uid], 'auth': 'normal',
            #                                                            'pos': 'replace'}))
            #         # 处理支援队列
            #         # 用于判断是否派出士兵增援的阈值，此战场敌我总血量相差越大，派出支援的概率越大
            #         threshold = pointsHealthSumMyself[i] / pointsHealthSumEnemy[i]
            #         for _, uid in enumerate(supporter):
            #             ut = self.myunits[uid]
            #             # 判断是否派出该士兵支援
            #             e = random.random()
            #             # 随机数小于阈值，不派出
            #             if e < threshold or sum(pointsSupportProp) == 0:
            #                 x, y = ut['POSITION'][0], ut['POSITION'][2]
            #                 enemy_id, distance = utils.get_closest(x, y, self.state['units_myself'])
            #                 self.set_target_objid(objid_list=[uid], targetObjID=enemy_id)
            #                 cmdThreads.append(threading.Thread(target=self.attack,
            #                                                    kwargs={'objid_list': [uid], 'auth': 'normal',
            #                                                            'pos': 'replace'}))
            #                 continue
            #             # 派出，判断支援哪个点，按照之前算出的各据点的支援权重，加权随机
            #             e = random.random()
            #             pos = 0
            #             for j in range(len(self.points_out)):
            #                 if pointsSupportProp[j] == 0:
            #                     continue
            #                 pos = j
            #                 if e > pointsSupportProp[j]:
            #                     e -= pointsSupportProp[j]
            #                 else:
            #                     break
            #             # 派出增援，命令其前往对应战场，并将其加入支援列表
            #             self.support_list[uid] = pos
            #             cmdThreads.append(threading.Thread(target=self.move,
            #                                                kwargs={'objid_list': [uid],
            #                                                        'destPos': self.points_out[pos],
            #                                                        'reachDist': 3, 'walkType': 'run'}))

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

                


