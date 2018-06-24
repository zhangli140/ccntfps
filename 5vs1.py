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
from control.assign_5vs1 import Assignment
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
        print(env.units_id)
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
                a, layers = dqn_e.choose_action(state[i], command_size, epsilon=var, enjoy=False)
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
    dqn = DQN_normal(resolution=resolution, command_size=command_size, index='0')
    # ----------------------------------init network------------------------------------------------
    point_list = [[125, -1, 95], [125, -1, 135], [165, -1, 95], [125, -1, 55], [85, -1, 95]]

    permutation_outside = list(permutations([0, 1, 2, 3]))
    permutation_inside = list(permutations([0, 1, 2, 3, 4]))

    print("begin init network.....")

    Priority = PriorityModel(4 * 9, len(permutation_outside), 100, 'prio_myself')
    Priority_e = PriorityModel(4 * 9, len(permutation_inside), 100, 'prio_enemy')
    dpt_out, dpt_in = get_dispatch_matrices(10)
    dispatcher = PriorityModel(4 * 9, len(dpt_out), 100, 'dpt_myself')
    dispatcher_e = PriorityModel(4 * 9, len(dpt_in), 100, 'dpt_enemy')
    attack = PriorityModel(4 * 9, 2, 100, 'attack')
    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        ckpt_p = tf.train.get_checkpoint_state(args.result + '/p_model')
        ckpt_pe = tf.train.get_checkpoint_state(args.result + '/p_model_e')
        ckpt_d = tf.train.get_checkpoint_state(args.result + '/d_model')
        ckpt_de = tf.train.get_checkpoint_state(args.result + '/d_model_e')
        ckpt_f = tf.train.get_checkpoint_state(args.result + '/f_model')
        if ckpt and ckpt.model_checkpoint_path:
            dqn.saver.restore(dqn.sess, ckpt.model_checkpoint_path)
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
    env.set_env(CONFIG.serverip, 5123, socket_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed, is_enemy=False, file_name='5vs1')
    agm = Assignment(env)
    print("finish init env!")


    recv = None
    _, s_enemy = env.reset()
    input()
    current_step = 0
    done = False
    fight = False

    print("finish reset env!")
    cant_f = 0
    while not fight:
        #cant_f += 1
        # formation algorithm
        prio_prob_e = Priority_e.calc_priority(s_enemy)
        prio_argm_e = np.argmax(prio_prob_e)
        if np.random.rand() < 0.05:
            prio_argm_e = np.random.randint(0, len(permutation_inside))
        prio_e = list(permutation_inside[prio_argm_e])

        s_sorted_e = dispatch_sort(s_enemy, prio_e)
        dpt_prob_e = dispatcher_e.calc_priority(s_sorted_e)

        dpt_argm_e_1 = np.argmax(dpt_prob_e)
        dpt_prob_e[dpt_argm_e_1] = -1
        dpt_argm_e_2 = np.argmax(dpt_prob_e)
        assign_p_e_1 = np.array(assign_sort(dpt_in[dpt_argm_e_1], prio_e))//2
        assign_p_e_1[-1] = 5 - sum(assign_p_e_1[:-1])
        assign_p_e_2 = np.array(assign_sort(dpt_in[dpt_argm_e_2], prio_e))//2
        assign_p_e_2[-1] = 5 - sum(assign_p_e_2[:-1])


        env.cpos_list1 = []
        env.cpos_list2 = []
        #print(assign_p_e_1)
        for i in range(5):
            for j in range(assign_p_e_1[i]):
                env.cpos_list1.append([point_list[i][0], point_list[i][2]])
            for j in range(assign_p_e_2[i]):
                env.cpos_list2.append([point_list[i][0], point_list[i][2]])

        copyagn1 = [assign_p_e_1[1], assign_p_e_1[2], assign_p_e_1[3], assign_p_e_1[4], assign_p_e_1[0]]
        copyagn2 = [assign_p_e_2[1], assign_p_e_2[2], assign_p_e_2[3], assign_p_e_2[4], assign_p_e_2[0]]
        env.assignment = np.array([copyagn1, copyagn2])

        env.stop = False
        env.pause = False
        print('assignment0:',env.assignment[0])
        threading.Thread(target=agm.Assign, args=([0, 0, 0, 0], env.assignment[0])).start()
        #print('finish assignment')
        while (len(env.client.strategy_select) < 1 and env.stop is False) or env.pause:
            time.sleep(1)
        env.stop = True
        if len(env.client.strategy_select):
            if env.client.strategy_select[0] == '0':
                cp = env.assignment[int(env.client.strategy_select[2])]
                copyagn = [cp[4], cp[0], cp[1], cp[2], cp[3]]
                print(copyagn)
                env.stop = False
                agm.Assign([0, 0, 0, 0], copyagn)
            else:
                fight = True
            env.client.strategy_select = ''
            print('finish_selected')
        if not fight:
            _, s_enemy_ = env.decay_feature()
            s_enemy = s_enemy_
            print('not fight')

        #编队 
        temp_team = {1:[],2:[],3:[],4:[],5:[]}
        inside_pos_list =[[125,-1,95],[125,-1,135],[165,-1,95],[125,-1,55],[85,-1,95]]
        for uid, u_data in env.states.items():
            if u_data['TEAM_ID'] > 0:
                x = u_data['POSITION'][0]
                y = u_data['POSITION'][2]
                for i in range(5):
                    if abs(x-inside_pos_list[i][0]) + abs(y-inside_pos_list[i][2]) < 12:
                        temp_team[i+1].append(uid)
                        break

        for team_id, objid_list in temp_team.items():
            print('teamid:', team_id)
            env.create_team(-1, objid_list, team_id)

    var = 0.2
    screen = env.reset_fight()
    screen_my, screen_enemy = screen['screen_my'], screen['screen_enemy']

    print('into mm')
    while not done:
        current_step += 1

        action, solLayers = get_action(screen_my, 'myself', env)
        print("action: {}".format(action))
        # action = [-1 for _ in range(len(env.units_id))]

        # action_e, solLayers_e = get_action(screen_enemy, 'enemy', env)
        # print("action_e: {}".format(action_e))
        action_e = [-1 for _ in range(len(env.units_e_id))]

        s_, _, done, _, _ = env.step([action, action_e])  # 执行完动作后的时间，time2

        screen_my_n = s_['screen_my']
        screen_enemy_n = s_['screen_enemy']

        screen_my = screen_my_n
        screen_enemy = screen_enemy_n