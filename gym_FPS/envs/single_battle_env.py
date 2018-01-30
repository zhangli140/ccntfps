# coding=utf-8
import time
import copy
import math
import numpy as np
from gym import spaces
from .. import utils

from . import FPS_env as fc
from .starcraft.Config import Config

DISTANCE_FACTOR = 20
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
        obs_low = np.zeros([1, 8])
        obs_high = (np.zeros([1, 8]) + 1) * 100
        return spaces.Box(np.array(obs_low), np.array(obs_high))


    def _reset(self):
        self.episodes += 1
        self.episode_steps = 0
        self.flag = 0
        self.new_episode()
        self.state = dict()
        self.state['units_myself'] = {}
        self.state['units_enemy'] = {}
        self.state['game_over'] = False
        self.state['win'] = False
        self.current_my_units = {}
        self.current_enemy_units = {}
        while len(self.states) == 0:
            time.sleep(0.1)          # 等待主角出现

        if np.random.random() < 0.5:
            flag = 0
        else:
            flag = 1

        names = ['队友', '敌人']
        self.add_obj(name="%s1"%names[flag], is_enemy=bool(flag), pos=[-212.5, -1, -26.7], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s2"%names[flag], is_enemy=bool(flag), pos=[-210.8, -1, -27.8], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s3"%names[flag], is_enemy=bool(flag), pos=[-208.9, -1, -28.2], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s4"%names[flag], is_enemy=bool(flag), pos=[-206.3, -1, -29.1], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s5"%names[flag], is_enemy=bool(flag), pos=[-204.3, -1, -29.5], leader_objid=-1, team_id=2-3*flag)

        self.add_obj(name="%s1"%names[1-flag], is_enemy=bool(1-flag), pos=[-190, -1, -13.5], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s2"%names[1-flag], is_enemy=bool(1-flag), pos=[-191, -1, -14.0], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s3"%names[1-flag], is_enemy=bool(1-flag), pos=[-193, -1, -13.6], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s4"%names[1-flag], is_enemy=bool(1-flag), pos=[-195, -1, -13.0], leader_objid=-1, team_id=2-3*flag)
        self.add_obj(name="%s5"%names[1-flag], is_enemy=bool(1-flag), pos=[-197, -1, -12.3], leader_objid=-1, team_id=2-3*flag)

        time.sleep(Config.sleeptime)
        self._make_feature()
        self.obs = self._make_observation()
        self.init_my_units = self.state['units_myself']
        unit_size = len(self.state['units_myself'])
        unit_size_e = len(self.state['units_enemy'])

        return self.obs, unit_size, unit_size_e




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
                degree = action[i][1] * np.pi
                distance = (action[i][2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself['POSITION'][0], myself['POSITION'][2])
                enemy_id, distance = utils.get_closest(x2, y2, self.state['units_enemy'])
                cmds.append([0, uid, enemy_id])
            else:
                # Move action
                if myself is None:
                    return cmds
                degree = action[i][1] * np.pi
                distance = (action[i][2] + 1) * DISTANCE_FACTOR
                x2, y2 = utils.get_position(degree, distance, myself['POSITION'][0], myself['POSITION'][2])
                cmds.append([1, uid, [x2, -1, y2]])
            i += 1
        print("commands send!", cmds)
        return cmds

    def die_fast(self):
        count_them = len(self.state['units_enemy'])
        if count_them != 0:
            cx, cy = utils.get_units_center(self.state['units_enemy'])
        else:
            cx, cy = utils.get_units_center(self.state['units_myself'])

        for uid, feats in self.state['units_myself'].items():
            print("commands", uid, cx, cy)
            print("position", feats['POSITION'][0], feats['POSITION'][1])
            self.move(objid_list=[uid], destPos=[cx, -1, cy], reachDist=3, walkType='run')
        self._make_feature()
        done = self.state['game_over']
        return done


    def _step(self, action):
        self.episode_steps += 1
        commands = self._make_commands(action)
        self.current_my_units = copy.deepcopy(self.state['units_myself'])
        self.current_enemy_units = copy.deepcopy(self.state['units_enemy'])
        print(len(self.state['units_myself']), len(self.state['units_enemy']))
        self.time1 = time.time()
        for i in range(len(commands)):
            print('POSITION', self.states[commands[i][1]]['POSITION'])
            if commands[i][0] == 0:
                unit = self.states[commands[i][2]]
                self.states[commands[i][1]]['LAST_CMD'] = [0, unit['POSITION'][0], unit['POSITION'][2]]
                if utils.get_dis(self.states[commands[i][1]]['POSITION'], self.states[commands[i][2]]['POSITION']) > 30:
                    self.move(objid_list=[commands[i][1]], destPos=self.states[commands[i][2]]['POSITION'],reachDist=3,walkType='run')
                else:
                    self.set_target_objid(objid_list=[commands[i][1]], targetObjID=commands[i][2])
                    self.attack(objid_list=[commands[i][1]], auth='normal', pos='replace')
            else:
                self.states[commands[i][1]]['LAST_CMD'] = [1, commands[i][2][0], commands[i][2][2]]
                self.move(objid_list=[commands[i][1]], destPos=commands[i][2], reachDist=3, walkType='run')
            self.states[commands[i][1]]['LAST_TIME'] = self.states[commands[i][1]]['TIME']
            self.states[commands[i][1]]['TIME'] = time.time()
            self.states[commands[i][1]]['LAST_POSITION_'] = self.states[commands[i][1]]['POSITION']
        for uid, ut in self.states.items():
            if ut['TEAM_ID'] < 0 and uid != 0 and ut['HEALTH'] > 0:
                self.states[uid]['LAST_TIME'] = self.states[uid]['TIME']
                self.states[uid]['TIME'] = time.time()
                self.states[uid]['LAST_POSITION_'] = self.states[uid]['POSITION']

        time.sleep(Config.sleeptime)
        self.time2 = time.time()
        self._make_feature()
        self.obs = self._make_observation()
        reward = self._compute_reward()
        print('reward', reward)
        done = self.state['game_over']
        unit_size = len(self.state['units_myself'])
        return self.obs, reward, done, unit_size




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
        observations = np.zeros([len(self.state['units_myself']) + len(self.state['units_enemy']), self.observation_space.shape[1]])  # [unit_size+enemy_size, 35]
 #       if (len(self.states) <= 11):
 #           print("right")

        count = 0
        for uid, ut in self.state['units_myself'].items():
      #      observations[count, 0] = uid
            observations[count, 0] = ut['HEALTH']/float(50)
            observations[count, 1] = ut['POSITION'][0]     #x
            observations[count, 2] = ut['POSITION'][2]     #y
            if 'LAST_CMD' not in ut.keys():
                observations[count, 5] = 0
                observations[count, 6] = 0
                observations[count, 7] = 0
            else:
                observations[count, 5] = ut['LAST_CMD'][1] / float(45)
                observations[count, 6] = ut['LAST_CMD'][2] / float(45)
                observations[count, 7] = ut['LAST_CMD'][0]

            if 'LAST_TIME' not in ut.keys() or 'LAST_POSITION_' not in ut.keys():
                observations[count, 3] = 0
                observations[count, 4] = 0
            else:
                observations[count, 3] = (ut['LAST_POSITION_'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
                # v-x
                observations[count, 4] = (ut['LAST_POSITION_'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
                # v-y
                #print(uid, ut['LAST_CMD'])
 #           observations[count, 11] = unit.type
 #           observations[count, 12] = unit.velocityX
 #           observations[count, 13] = unit.velocityY
            count += 1
        for uid, ut in self.state['units_enemy'].items():
            observations[count, 0] = ut['HEALTH'] / float(50)
            observations[count, 1] = ut['POSITION'][0]  # x
            observations[count, 2] = ut['POSITION'][2]
            if 'LAST_CMD' not in ut.keys():
                observations[count, 5] = 0
                observations[count, 6] = 0
                observations[count, 7] = 0
            else:
                observations[count, 5] = ut['LAST_CMD'][1] / float(45)
                observations[count, 6] = ut['LAST_CMD'][2] / float(45)
                observations[count, 7] = ut['LAST_CMD'][0]

            if 'LAST_TIME' not in ut.keys():
                observations[count, 3] = 0
                observations[count, 4] = 0
            else:
                observations[count, 3] = (ut['LAST_POSITION_'][0] - ut['POSITION'][0]) / float(ut['TIME'] - ut['LAST_TIME'])
                # v-x
                observations[count, 4] = (ut['LAST_POSITION_'][2] - ut['POSITION'][2]) / float(ut['TIME'] - ut['LAST_TIME'])
                print('TIME', ut['TIME'], ut['LAST_TIME'])
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

    def _unatural_reward(self):
        tmp_my = 0
        tmp_enemy = 0
        if len(self.current_my_units) == 0 or len(self.current_enemy_units) == 0:
            return None

        for uid, ut in self.current_my_units.items():      #action执行前
            if uid in self.init_my_units:
                tmp_my += ut['HEALTH']

        for uid, ut in self.current_enemy_units.items():
            if uid not in self.state['units_enemy']:
                tmp_enemy += ut['HEALTH']
            else:
                tmp_enemy += ut['HEALTH'] - self.state['units_enemy'][uid]['HEALTH']

        tmp_enemy /= len(self.current_enemy_units)
        tmp_my /= len(self.current_my_units)
        return tmp_enemy - tmp_my



