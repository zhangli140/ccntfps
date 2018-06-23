# coding=utf8

import numpy as np
import gym_FPS
from gym_FPS.envs.starcraft.Config import Config
import argparse
import pickle
from gym_FPS.envs.starcraft.model import DDPG, DQN, DQN_normal
import tensorflow as tf
import pylab
import time
import os, gym
from gym_FPS.utils import *
import threading
from control.assign import Assignment
from control.priority import PriorityModel
from itertools import permutations
from control.dispatch import *
from socket import *
import Memory
import scipy.misc
import win32gui

# action:(attack or move, degree, distance)
# state:()
#  * hit points, cooldown, ground range, is enemy, degree, distance (myself)
#  * hit points, cooldown, ground range, is enemy (enemy)
CONFIG = Config()
MEMORY_CAPACITY = CONFIG.memory_capacity
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='server ip', default=CONFIG.serverip)
parser.add_argument('--port', help='server port', default=CONFIG.serverport)
parser.add_argument('--result', help='result', default='result')
args = parser.parse_args()

if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.result + '/model'):
    os.mkdir(args.result + '/model')
if not os.path.exists(args.result + '/model_e'):
    os.mkdir(args.result + '/model_e')
if not os.path.exists(args.result + '/p_model'):
    os.mkdir(args.result + '/p_model')
if not os.path.exists(args.result + '/p_model_e'):
    os.mkdir(args.result + '/p_model_e')
if not os.path.exists(args.result + '/d_model'):
    os.mkdir(args.result + '/d_model')
if not os.path.exists(args.result + '/d_model_e'):
    os.mkdir(args.result + '/d_model_e')
if not os.path.exists(args.result + '/f_model'):
    os.mkdir(args.result + '/f_model')

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_round = 1
batch_size = 16

command_size = 9
resolution = (255, 255)



def wf(str, flag):
    if flag == 0:
        filepath = args.result + '/win.txt'
    else:
        filepath = args.result + '/reward.txt'
    F_battle = open(filepath, 'a')
    F_battle.write(str + '\n')
    F_battle.close()

def get_action(state, flag, env, use_rule=False):
    action = []
    s = []
    # command_size = 14
    solLayers = []

    if flag == 'myself':
        for i in range(len(env.units_id)):
            uid = env.units_id[i]
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0:
                if use_rule:
                    action.append(8)
                    solLayers.append(dict())
                    continue
                a, layers = dqn.choose_action(state[i], command_size, epsilon=var, enjoy=False)
                action.append(a)
                solLayers.append(layers)
            else:
                action.append(-1)  # -1 means invalid action
                solLayers.append(dict())

    else:
        for i in range(len(env.units_e_id)):
            uid = env.units_e_id[i]
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0:
                if use_rule:
                    action.append(8)
                    solLayers.append(dict())
                    continue
                a, layers = dqn.choose_action(state[i], command_size, epsilon=var, enjoy=False)
                action.append(a)
                solLayers.append(layers)
            else:
                action.append(-1)  # -1 means invalid action
                solLayers.append(dict())

    return action, solLayers

def wfTo(filepath, str):
    F_battle = open(filepath, 'a')
    F_battle.write(str + '\n')
    F_battle.close()

if __name__ == '__main__':
    buffer_size = 10000
    point_list = [[125, -1, 175], [205, -1, 95], [125, -1, 15], [45, -1, 95]]

    permutation_outside = list(permutations([0, 1, 2, 3]))
    permutation_inside = list(permutations([0, 1, 2, 3, 4]))
    # ----------------------------------init network------------------------------------------------
    print("begin init network.....")
    dqn = DQN_normal(resolution=resolution, command_size=command_size, index='0')
    # dqn_e = DQN_normal(resolution=resolution, command_size=command_size, index='0')

    Priority = PriorityModel(4 * 9, len(permutation_outside), 100, 'prio_myself')
    Priority_e = PriorityModel(4 * 9, len(permutation_inside), 100, 'prio_enemy')
    dpt_out, dpt_in = get_dispatch_matrices(10)
    dispatcher = PriorityModel(4 * 9, len(dpt_out), 100, 'dpt_myself')
    dispatcher_e = PriorityModel(4 * 9, len(dpt_in), 100, 'dpt_enemy')
    attack = PriorityModel(4 * 9, 2, 100, 'attack')
    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        # ckpt_e = tf.train.get_checkpoint_state(args.result + '/model_e')
        ckpt_p = tf.train.get_checkpoint_state(args.result + '/p_model')
        ckpt_pe = tf.train.get_checkpoint_state(args.result + '/p_model_e')
        ckpt_d = tf.train.get_checkpoint_state(args.result + '/d_model')
        ckpt_de = tf.train.get_checkpoint_state(args.result + '/d_model_e')
        ckpt_f = tf.train.get_checkpoint_state(args.result + '/f_model')
        if ckpt and ckpt.model_checkpoint_path:
            dqn.saver.restore(dqn.sess, ckpt.model_checkpoint_path)
        # if ckpt_e and ckpt_e.model_checkpoint_path:
        #     dqn_e.saver.restore(dqn_e.sess, ckpt.model_checkpoint_path)
        if ckpt_p and ckpt_p.model_checkpoint_path:
            Priority.saver.restore(Priority.sess, ckpt_p.model_checkpoint_path)
        if ckpt_pe and ckpt_pe.model_checkpoint_path:
            Priority_e.saver.restore(Priority_e.sess, ckpt_pe.model_checkpoint_path)
        if ckpt_d and ckpt_d.model_checkpoint_path:
            dispatcher.saver.restore(dispatcher.sess, ckpt_d.model_checkpoint_path)
        if ckpt_de and ckpt_de.model_checkpoint_path:
            dispatcher_e.saver.restore(dispatcher_e.sess, ckpt_de.model_checkpoint_path)
        if ckpt_f and ckpt_f.model_checkpoint_path:
            attack.saver.restore(attack.sess, ckpt_f.model_checkpoint_path)

    print("finish init network")
    # ----------------------------------init env------------------------------------------
    env = gym.make('FPSDouble-v0')
    print("begin init env....")
    env.set_env(args.ip, 5123, socket_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed, is_enemy=False, is_5player=True)

    agm = Assignment(env)    # env.restart(port=args.port)
    env.assignment = np.zeros((3, 4), dtype=np.int32)    

    env.seed(123)
    print("finish init env!")

    episodes = 0
    battles_won = 0
    battles_won_total = 0
    var = 0.2
    win_rate = []

    s, _ = env.reset()
    input()#等待client启动


    current_step = 0
    done = False
    rewards = []
    epi_flag = True
    tmp_loss = 0
    reward = 0
    cumulative_reward = 0
    fight = False
    cant_f = 0
    print("finish reset env!")
    
    while not fight:
        cant_f += 1
        # formation algorithm
        prio_prob = Priority.calc_priority(s)
        prio_argm = np.argmax(prio_prob)
        if np.random.uniform(0, 1) < 0.05:
            prio_argm = np.random.randint(0, len(permutation_outside))
        prio = list(permutation_outside[prio_argm])

        s_sorted = dispatch_sort(s, prio)
        dpt_prob = dispatcher.calc_priority(s_sorted)

        dpt_argm_1 = np.argmax(dpt_prob)
        dpt_prob[dpt_argm_1] = -1
        dpt_argm_2 = np.argmax(dpt_prob)
        assign_p_1 = np.array(assign_sort(dpt_out[dpt_argm_1], prio))//2
        assign_p_1[-1] = 5 - sum(assign_p_1[:-1])
        assign_p_2 = np.array(assign_sort(dpt_out[dpt_argm_2], prio))//2
        assign_p_2[-1] = 5 - sum(assign_p_2[:-1])

        env.assignment = np.array([assign_p_1, assign_p_2, env.assignment[2]])
        print(env.assignment[2])
        env.cpos_list1 = []
        env.cpos_list2 = []
        for i in range(4):
            for j in range(assign_p_1[i]):
                env.cpos_list1.append([point_list[i][0], point_list[i][2]])
            for j in range(assign_p_2[i]):
                env.cpos_list2.append([point_list[i][0], point_list[i][2]])
        print(229)
        #agm.Assign(assign_p_1, [0, 0, 0, 0, 0])
        env.stop = False
        threading.Thread(target=agm.Assign, args=(assign_p_1, [0, 0, 0, 0, 0])).start()
        while len(env.client.strategy_select) < 1 and env.stop is False:
            time.sleep(1)
        print(235)
        
        if len(env.client.strategy_select):
            if env.client.strategy_select[0] == '0':
                agm.Assign(env.assignment[int(env.client.strategy_select[2])], [0, 0, 0, 0, 0])
            else:
                fight = True
            env.client.strategy_select = ''
        env.stop = True
        if not fight:
            s_, _ = env.decay_feature()

            fight_prob = attack.calc_priority(s_)
            fight = np.argmax(fight_prob)
            if np.random.uniform(0, 1) < 0.05:
                fight = np.random.randint(0, 2)
            if cant_f == 1:
                fight = False
            elif cant_f > 3:
                fight = True


        s = s_
        #编队
        temp_team = {1:[],2:[],3:[],4:[]}
        outside_pos_list=[[125,-1,175],[185,-1,155],[205,-1,95],[185,-1,35],[125,-1,15],[55,-1,35],[45,-1,95],[55,-1,155]]
        for uid, u_data in env.states.items():
            if u_data['TEAM_ID'] > 0:
                x = u_data['POSITION'][0]
                y = u_data['POSITION'][2]
                for i in range(4):
                    if abs(x-outside_pos_list[i*2][0]) + abs(y-outside_pos_list[i*2][2]) < 12:
                        temp_team[i+1].append(uid)
                        break

        for team_id, objid_list in temp_team.items():
            env.create_team(-1, objid_list, team_id)





    screen = env.reset_fight()
    screen_my, screen_enemy = screen['screen_my'], screen['screen_enemy']
    
    
    while not done:

        current_step += 1

        action, solLayers = get_action(screen_my, 'myself', env)
        print("action: {}".format(action))

        action_e, solLayers_e = get_action(screen_enemy, 'enemy', env, use_rule=True)
        print("action_e: {}".format(action_e))

        s_, reward, done, unit_size_, unit_size_e_ = env.step([action, action_e])  # 执行完动作后的时间，time2
        print("reward: {}".format(reward))

        screen_my_n = s_['screen_my']
        screen_enemy_n = s_['screen_enemy']

        screen_my = screen_my_n
        screen_enemy = screen_enemy_n


    if epi_flag:
        episodes += 1
        R = env.formation_reward()
        if R > 0:
            battles_won += 1
        if episodes % 50 == 0:
            win_rate.append(battles_won / 50)
            print('episodes: ', episodes, 'win_rate: ', battles_won / 50, 'cum_win_rate:', np.mean(win_rate))
            battles_won = 0
            env.restart()

