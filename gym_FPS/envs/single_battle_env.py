# coding=utf-8
from __future__ import division
import numpy as np
import math
from gym import spaces
from .. import utils 

from . import FPS_env as fc
import time
from .starcraft.Config import *
import copy

DISTANCE_FACTOR = 16
ENEMY = 1
MYSELF = 0


class SingleBattleEnv(fc.FPSEnv):
    def __init__(self, max_episode_steps=20000):
        super(SingleBattleEnv, self).__init__()

        '''
        配置env
        '''
        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.current_my_units = {}
        self.current_enemy_units = {}
        self.max_episode_steps = max_episode_steps
        self.episodes = 0
        self.episode_wins = 0
        self.episode_steps = 0
        self.init_my_units = {}
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()
        self.flag = True
        self.time1 = 0
        self.time2 = 0

    def _action_space(self):
        action_low = [-1.0, -math.pi/2, -1.0]
        action_high = [1.0, math.pi/2, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        obs_low = np.zeros([1,10])
        obs_high = (np.zeros([1,10])+1)*100
        return spaces.Box(np.array(obs_low), np.array(obs_high))


    def _reset(self):
        self.episodes += 1
        self.episode_steps = 0
        self.flag = 0
        self.new_episode()
        self.state['game_over'] = False
        self.state['win'] =False
        while(len(self.states) == 0):
            time.sleep(0.1)          # 等待主角出现

        self.add_obj(name="敌人1", is_enemy=True, pos=[-212.5, -1, -26.7], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人2", is_enemy=True, pos=[-210.8, -1, -27.8], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人3", is_enemy=True, pos=[-208.9, -1, -28.2], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人4", is_enemy=True, pos=[-206.3, -1, -29.1], leader_objid=-1, team_id=-1)
        self.add_obj(name="敌人5", is_enemy=True, pos=[-204.3, -1, -29.5], leader_objid=-1, team_id=-1)

        self.add_obj(name="队友1", is_enemy=False, pos=[-190, -1, -13.5], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友2", is_enemy=False, pos=[-191, -1, -14.0], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友3", is_enemy=False, pos=[-193, -1, -13.6], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友4", is_enemy=False, pos=[-195, -1, -13.0], leader_objid=-1, team_id=-1)
        self.add_obj(name="队友5", is_enemy=False, pos=[-197, -1, -12.3], leader_objid=-1, team_id=-1)

        time.sleep(Config.sleeptime)
        self._make_feature()
        self.obs =self._make_observation()
        self.init_my_units = self.state['units_myself']
        unit_size = len(self.state['units_myself'])
        unit_size_e = len(self.state['units_enemy'])
        self.add_observer([-220, -1, 20], 2000)
        return self.obs,unit_size,unit_size_e




    def _make_commands(self, action):
        cmds = []
        self.current_my_units = self.state['units_myself']
        self.current_enemy_units = self.state['units_enemy']
        if self.state is None or (len(action) == 0):
            return cmds
        if len(action) is not len(self.state['units_myself']):
            return cmds
        i = 0
        for uid, ut in self.state['units_myself'].items():
            myself = ut
            if action[i][0] < 0:
                # Attack action
                if myself is None:
                    return cmds
                degree = action[i][1]
                distance = (action[i][2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself['POSITION'][0], myself['POSITION'][2])
                enemy_id, distance = utils.get_closest(x2, y2, self.state['units_enemy'])
                cmds.append([0,uid,enemy_id])
            else:
                # Move action
                if myself is None:
                    return cmds
                degree = action[i][1]
                distance = (action[i][2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself['POSITION'][0], myself['POSITION'][2])
                cmds.append([1, uid, [x2, -1, y2]])
            i += 1
        # print "commands send!"
        return cmds

    def die_fast(self):
        count_them = len(self.state['units_enemy'])
        actions = []
        if count_them != 0:
            cx, cy = utils.get_units_center(self.state['units_enemy'])
        else:
            cx, cy = utils.get_units_center(self.state['units_myself'])

        for uid, feats in self.state['units_myself'].items():
            self.move(objid_list=[uid], destPos=[[cx,-1,cy]], reachDist=0, walkType='run')
        self._make_feature()
        done = self.state['game_over']
        return done


    def _step(self, action):
        self.episode_steps += 1
        action = action.copy()####防止影响store
        action[-1] *= np.pi#####degree范围不应该是[-1,1]
        commands = self._make_commands(action)
        print(commands)
        self.current_my_units = copy.deepcopy(self.state['units_myself'])
        self.current_enemy_units = copy.deepcopy(self.state['units_enemy'])
        self.time1 = time.time()
        print('time1',self.time1,'time2',self.time2,"time gap", self.time1 - self.time2)     #第一次动作执行完到第二次动作开始
        for i in range(len(commands)):
            if commands[i][0]==0:
                unit = self.states[commands[i][2]]
                self.states[commands[i][1]]['LAST_CMD']=[0, unit['POSITION'][0], unit['POSITION'][2]]
                self.set_target_objid(objid_list=[commands[i][1]],targetObjID=commands[i][2])
                self.attack(objid_list=[commands[i][1]],auth='normal',pos='replace')
            else:
                self.states[commands[i][1]]['LAST_CMD']=[1, commands[i][2][0], commands[i][2][2]]
                self.move(objid_list=[commands[i][1]],destPos=[commands[i][2]],reachDist=0,walkType='run')



        time.sleep(Config.sleeptime)
        self.time2 = time.time()
        self._make_feature()
        self.obs = self._make_observation()
        reward = self._compute_reward()
        print('reward',reward)
        done = self.state['game_over']
        unit_size = len(self.state['units_myself'])
        print(unit_size)
        return self.obs,reward,done,unit_size




    def _make_feature(self):
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        for uid, ut in self.states.items():
            if ut['TEAM_ID'] > 0 and uid != 0 and ut['HEALTH'] > 0:
                self.state['units_myself'][uid] = self.states[uid]
            elif uid != 0 and ut['HEALTH'] > 0:
                self.state['units_enemy'][uid] = self.states[uid]
        if len(self.state['units_myself']) == 0 or len(self.state['units_enemy']) == 0:
            self.state['game_over'] = True
            if len(self.state['units_myself']) > 0:
                self.state['battle_won'] = True
            else:
                self.state['battle_won'] = False





    def _make_observation(self):
        observations = np.zeros([len(self.states) - 1,self.observation_space.shape[1]])  # [unit_size+enemy_size, 35]
        if (len(self.states) <= 11):
            print("right")

        count = 0
        for uid, ut in self.state['units_myself'].items():
            observations[count, 0] = uid
            observations[count, 1] = ut['HEALTH']/float(50)
            observations[count, 2] = ut['POSITION'][0]     #x
            observations[count, 3] = ut['POSITION'][2]     #y
            observations[count, 4] = (ut['LAST_POSITION'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
            observations[count, 5] = (ut['LAST_POSITION'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
            if 'LAST_CMD' not in ut.keys():
                observations[count, 6] = 0
                observations[count, 7] = 0
                observations[count, 8] = 0
            else:
                observations[count, 6] = ut['LAST_CMD'][1] / float(45)
                observations[count, 7] = ut['LAST_CMD'][2] / float(45)
                observations[count, 8] = ut['LAST_CMD'][0]
                #print(uid, ut['LAST_CMD'])
 #           observations[count, 11] = unit.type
 #           observations[count, 12] = unit.velocityX
 #           observations[count, 13] = unit.velocityY
            count += 1
        for uid,ut in self.state['units_enemy'].items():
            observations[count, 0] = uid
            observations[count, 1] = ut['HEALTH'] / float(50)
            observations[count, 2] = ut['POSITION'][0]  # x
            observations[count, 3] = ut['POSITION'][2]
            observations[count, 4] = (ut['LAST_POSITION'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
            observations[count, 5] = (ut['LAST_POSITION'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
            if 'LAST_CMD' not in ut.keys():
                observations[count, 6] = 0
                observations[count, 7] = 0
                observations[count, 8] = 0
            else:
                observations[count, 6] = ut['LAST_CMD'][1] / float(45)
                observations[count, 7] = ut['LAST_CMD'][2] / float(45)
                observations[count, 8] = ut['LAST_CMD'][0]
            count += 1
        return np.asarray(observations)

    def _compute_reward(self):
        tmp_my = 0
        tmp_enemy = 0
        if len(self.current_my_units) == 0 or len(self.current_enemy_units) == 0:
            return None

        for uid, ut in self.current_my_units.items():      #action执行前
            if uid in self.init_my_units and uid not in self.state['units_myself']:
                tmp_my += ut['HEALTH']
            elif uid in self.state['units_myself']:
                tmp_my += ut['HEALTH'] - self.state['units_myself'][uid]['HEALTH']

        for uid, ut in self.current_enemy_units.items():
            if uid not in self.state['units_enemy']:
                tmp_enemy += ut['HEALTH']
            else:
                tmp_enemy += ut['HEALTH'] - self.state['units_enemy'][uid]['HEALTH']

        tmp_enemy /= len(self.current_enemy_units)
        tmp_my /= len(self.current_my_units)
        return tmp_enemy - tmp_my


