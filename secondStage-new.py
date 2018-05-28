#coding=utf8

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
parser.add_argument('--result', help='result', default='../result_dqn_vs_rule_ri_255x255_s2_test25_new_reward')
args = parser.parse_args()

# Switches
np.set_printoptions(threshold=np.inf)

printMiddleResult = False
printAction = True
printWeight = False
train = False

if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.result + '/model'):
    os.mkdir(args.result + '/model')
if not os.path.exists(args.result + '/model_e'):
    os.mkdir(args.result + '/model_e')
if printMiddleResult and not os.path.exists(args.result + '/hidden_results'):
    os.mkdir(args.result + '/hidden_results')
hidden_results_dir = args.result + '/hidden_results'
if printWeight and not os.path.exists(args.result + '/weights'):
    os.mkdir(args.result + '/weights')
weights_dir = args.result + '/weights'

os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

def get_action(state, flag, env):
    action = []
    s = []
    # command_size = 14
    solLayers = []


    if flag == 'myself':
        for i in range(len(env.units_id)):
            uid = env.units_id[i]
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0:
                a, layers = dqn.choose_action(state[i], command_size, epsilon=var, enjoy=False)
                action.append(a)
                solLayers.append(layers)
            else:
                action.append(-1) # -1 means invalid action
                solLayers.append(dict())
        
    else:
        for i in range(len(env.units_e_id)):
            uid = env.units_e_id[i]
            ut = env.myunits[uid]
            if ut['HEALTH'] > 0 :
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

def outputLayers(hr_epi_step_dir, solLayers):
    max_conv1, max_conv2, max_conv3 = None, None, None
    min_conv1, min_conv2, min_conv3 = None, None, None
    for i, it in enumerate(solLayers):
        if 'conv1' in it.keys():
            if max_conv1 is None:
                max_conv1 = np.max(np.array(it['conv1']))
            else:
                max_conv1 = max(max_conv1, np.max(np.array(it['conv1'])))
            if min_conv1 is None:
                min_conv1 = np.min(np.array(it['conv1']))
            else:
                min_conv1 = min(min_conv1, np.min(np.array(it['conv1'])))
        if 'conv2' in it.keys():
            if max_conv2 is None:
                max_conv2 = np.max(np.array(it['conv2']))
            else:
                max_conv2 = max(max_conv2, np.max(np.array(it['conv2'])))
            if min_conv2 is None:
                min_conv2 = np.min(np.array(it['conv2']))
            else:
                min_conv2 = min(min_conv2, np.min(np.array(it['conv2'])))
        if 'conv3' in it.keys():
            if max_conv3 is None:
                max_conv3 = np.max(np.array(it['conv3']))
            else:
                max_conv3 = max(max_conv3, np.max(np.array(it['conv3'])))
            if min_conv3 is None:
                min_conv3 = np.min(np.array(it['conv3']))
            else:
                min_conv3 = min(min_conv3, np.min(np.array(it['conv3'])))

    for i, it in enumerate(solLayers):
        # if 'input' in it.keys():
        #     if not os.path.exists(hr_epi_step_dir + '/input'):
        #         os.mkdir(hr_epi_step_dir + '/input')
        #     input_dir = hr_epi_step_dir + '/input'
        #     img = np.array(it['input']).copy()
        #     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
        #     img = img[:, :, 0]
        #     scipy.misc.imsave(input_dir + '/sol#{}.png'.format(i), img)
        #
        # if 'conv1' in it.keys():
        #     if not os.path.exists(hr_epi_step_dir + '/conv1'):
        #         os.mkdir(hr_epi_step_dir + '/conv1')
        #     conv1_dir = hr_epi_step_dir + '/conv1'
        #     conv1 = np.array(it['conv1']).copy()
        #     for nFilters in range(conv1.shape[2]):
        #         img = conv1[:, :, nFilters].copy()
        #         if not os.path.exists(conv1_dir + '/filter#{}'.format(nFilters)):
        #             os.mkdir(conv1_dir + '/filter#{}'.format(nFilters))
        #         filter_dir = conv1_dir + '/filter#{}'.format(nFilters)
        #         if max_conv1 == min_conv1:
        #             img[:, :] = 0
        #         else:
        #             img = (img - min_conv1) / (max_conv1 - min_conv1) * 255
        #         scipy.misc.imsave(filter_dir + '/sol#{}.png'.format(i), img)
        #
        # if 'conv2' in it.keys():
        #     if not os.path.exists(hr_epi_step_dir + '/conv2'):
        #         os.mkdir(hr_epi_step_dir + '/conv2')
        #     conv2_dir = hr_epi_step_dir + '/conv2'
        #     conv2 = np.array(it['conv2']).copy()
        #     for nFilters in range(conv2.shape[2]):
        #         img = conv2[:, :, nFilters].copy()
        #         if not os.path.exists(conv2_dir + '/filter#{}'.format(nFilters)):
        #             os.mkdir(conv2_dir + '/filter#{}'.format(nFilters))
        #         filter_dir = conv2_dir + '/filter#{}'.format(nFilters)
        #         if max_conv2 == min_conv2:
        #             img[:, :] = 0
        #         else:
        #             img = (img - min_conv2) / (max_conv2 - min_conv2) * 255
        #         scipy.misc.imsave(filter_dir + '/sol#{}.png'.format(i), img)
        #
        # if 'conv3' in it.keys():
        #     if not os.path.exists(hr_epi_step_dir + '/conv3'):
        #         os.mkdir(hr_epi_step_dir + '/conv3')
        #     conv3_dir = hr_epi_step_dir + '/conv3'
        #     conv3 = np.array(it['conv3']).copy()
        #     for nFilters in range(conv3.shape[2]):
        #         img = conv3[:, :, nFilters].copy()
        #         if not os.path.exists(conv3_dir + '/filter#{}'.format(nFilters)):
        #             os.mkdir(conv3_dir + '/filter#{}'.format(nFilters))
        #         filter_dir = conv3_dir + '/filter#{}'.format(nFilters)
        #         if max_conv3 == min_conv3:
        #             img[:, :] = 0
        #         else:
        #             img = (img - min_conv3) / (max_conv3 - min_conv3) * 255
        #         scipy.misc.imsave(filter_dir + '/sol#{}.png'.format(i), img)
        #
        # if 'flatten' in it.keys():
        #     wfTo(hr_epi_step_dir + '/flatten.txt', '#{}\n{}'.format(i, it['flatten']))
        # else:
        #     wfTo(hr_epi_step_dir + '/flatten.txt', '#{}\n{}'.format(i, 'None'))
        #
        # if 'fc1' in it.keys():
        #     wfTo(hr_epi_step_dir + '/fc1.txt', '#{}\n{}'.format(i, it['fc1']))
        # else:
        #     wfTo(hr_epi_step_dir + '/fc1.txt', '#{}\n{}'.format(i, 'None'))
        #
        # if 'fc2' in it.keys():
        #     wfTo(hr_epi_step_dir + '/fc2.txt', '#{}\n{}'.format(i, it['fc2']))
        # else:
        #     wfTo(hr_epi_step_dir + '/fc2.txt', '#{}\n{}'.format(i, 'None'))
        #
        # if 'fc3' in it.keys():
        #     wfTo(hr_epi_step_dir + '/fc3.txt', '#{}\n{}'.format(i, it['fc3']))
        # else:
        #     wfTo(hr_epi_step_dir + '/fc3.txt', '#{}\n{}'.format(i, 'None'))

        if 'output' in it.keys():
            wfTo(hr_epi_step_dir + '/q.txt', '#{}\n{}'.format(i, it['output']))
        else:
            wfTo(hr_epi_step_dir + '/q.txt', '#{}\n{}'.format(i, 'None'))

def printGrad(filepath, grad):
    f = open(filepath + '/grad_conv1_kernel.txt', 'a')
    f.write(str(grad[0]) + '\n')
    f.close()

    f = open(filepath + '/grad_conv1_bias.txt', 'a')
    f.write(str(grad[1]) + '\n')
    f.close()

    f = open(filepath + '/grad_conv2_kernel.txt', 'a')
    f.write(str(grad[2]) + '\n')
    f.close()

    f = open(filepath + '/grad_conv2_bias.txt', 'a')
    f.write(str(grad[3]) + '\n')
    f.close()

    # f = open(filepath + '/grad_conv3_kernel.txt', 'a')
    # f.write(str(grad[4]) + '\n')
    # f.close()
    #
    # f = open(filepath + '/grad_conv3_bias.txt', 'a')
    # f.write(str(grad[5]) + '\n')
    # f.close()

    # f = open(filepath + '/dense1_kernel.txt', 'a')
    # f.write(str(output[6]) + '\n')
    # f.close()

    f = open(filepath + '/grad_dense2_kernel.txt', 'a')
    f.write(str(grad[5]) + '\n')
    f.close()

    f = open(filepath + '/grad_dense3_kernel.txt', 'a')
    f.write(str(grad[6]) + '\n')
    f.close()

    f = open(filepath + '/grad_dense4_kernel.txt', 'a')
    f.write(str(grad[7]) + '\n')
    f.close()

    f = open(filepath + '/grad_dense4_bias.txt', 'a')
    f.write(str(grad[8]) + '\n')
    f.close()



if __name__ == '__main__':
    buffer_size = 10000
    #----------------------------------init env------------------------------------------
    env = gym.make('FPSSingle-v0')
    print("begin init env....")
    env.set_env(args.ip, args.port, client_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed)
    # env.restart(port=args.port)
    env.seed(123)
    print("finish init env!")
    #----------------------------------init memory--------------------------------------------------
    replay_buffer = Memory.ReplayMemory_with_dead_index(capacity=buffer_size, resolution=resolution)
    #----------------------------------init network------------------------------------------------
    print("begin init network.....")
    dqn = DQN_normal(resolution=resolution, command_size = command_size,index = '0')

    if CONFIG.load == 1:
        print("OLD VAR LOADED")
        ckpt = tf.train.get_checkpoint_state(args.result + '/model')
        if ckpt and ckpt.model_checkpoint_path:
            print("OLD VARS!")
            dqn.saver.restore(dqn.sess, ckpt.model_checkpoint_path)
    print("finish init network")

    if printWeight:
        if not os.path.exists(weights_dir + '/learned_0'):
            os.mkdir(weights_dir + '/learned_0')
        dqn.saveParaTo(weights_dir + '/learned_0')


    episodes = CONFIG.episode
    battles_won = 0
    battles_won_total = 0

    var = 0.3 * (0.9999 ** CONFIG.episode)  # control exploration
    if not train:
        var = 0.2
    #print('var is ', var)
    win_rate = {}

    nTrain = 0

    while episodes < 10000:
        if printMiddleResult and not os.path.exists(hidden_results_dir + '/episode#' + str(episodes)):
            os.mkdir(hidden_results_dir + '/episode#' + str(episodes))
            hr_epi_dir = hidden_results_dir + '/episode#' + str(episodes)

        s, unit_size, e_unit_size = env.reset()
        screen_my = s
        current_step = 0
        done = False
        rewards = []
        epi_flag = True
        tmp_loss = 0
        reward = 0

        cumulative_reward = 0
        unatural_flag = False
        print("finish reset env!")

        while not done:
            # 检测游戏是否挂掉了
            if not win32gui.FindWindow(0, 'Fps[服务器端:10000]'):
                env.__del__()
                env = gym.make('FPSSingle-v0')
                env.set_env(args.ip, args.port, client_DEBUG=False, env_DEBUG=False, speedup=CONFIG.speed)
                env.seed(123)
                print("restart")
                epi_flag = False
                break

            current_step += 1

            if printMiddleResult and not os.path.exists(hr_epi_dir + '/step#' + str(current_step)):
                os.mkdir(hr_epi_dir + '/step#' + str(current_step))
                hr_epi_step_dir = hr_epi_dir + '/step#' + str(current_step)

            action, solLayers = get_action(screen_my, 'myself', env)
            print("action: {}".format(action))
            if printAction:
                wfTo(args.result + '/action.txt', 'episode: {} step: {}\n{}\n\n'.format(episodes, current_step, str(action)))

            s_, reward, done, unit_size_ = env.step(action)#执行完动作后的时间，time2
            print("reward: {}".format(reward))
            wfTo(args.result + '/reward.txt', 'episode: {} step: {}\n{}\n\n'.format(episodes, current_step, str(reward)))

            # keep middle result
            if printMiddleResult:
                outputLayers(hr_epi_step_dir=hr_epi_step_dir, solLayers=solLayers)
                wfTo(hr_epi_dir + '/action.txt', 'episode: {} step: {}\n{}\n\n'.format(episodes, current_step, str(action)))

            # time.sleep(0.8)
            screen_my_n = s_

            if reward is not None:
                rewards.append(reward)
                for i in range(len(env.units_e_id)):
                    if(action[i]!=-1):
                        replay_buffer.add_transition(screen_my[i], action[i], screen_my_n[i], 0, rewards[-1],command_size,command_size,[])


            if train:
                var *= 0.9999
            screen_my = screen_my_n

            # current_step += 1
            if reward is not None:
                cumulative_reward += reward
        if epi_flag:
            episodes += 1
            if bool(env.state['battle_won']):
                battles_won += 1
                battles_won_total += 1

            # wf(str(battles_won) + '\n' + str(episodes) + '\n', 0)
            # wf(str(np.mean(rewards)) + '\n' + str(episodes) + '\n', 1)
            # wfTo(args.result + '/reward.txt', 'episodes: {}\nmean reward: {}\n\n'.format(episodes, np.mean(rewards)))
            wfTo(args.result + '/win.txt', 'episodes: {}\nwin in 50: {}\nwin rate: {}\n\n'.format(episodes, battles_won, battles_won / (episodes % 50 if episodes % 50 != 0 else 50) * 100))
            wfTo(args.result + '/win_total.txt', 'episodes: {}\nwin: {}\nwin rate: {}\n\n'.format(episodes, battles_won_total, battles_won_total / episodes * 100))

            print('episodes:', episodes, ', win:', battles_won, ', mean_reward:', np.mean(rewards))
            if episodes % CONFIG.episode_to_reset_win == 0:
                win_rate[episodes] = battles_won / CONFIG.episode_to_reset_win
                print('win rate:', win_rate[episodes])
                battles_won = 0

            #--------------------------begin train---------------------------------
            if train:
                print("begin train!")
                replay_buffer.shuffle()
                # state_action_r, action_r, state_action_next_r, isterminal_r, rewards_r, command_size_r, command_size_next_r, dead_e_index_r = replay_buffer.get_sample(
                #     min(batch_size, replay_buffer.size))
                # loss, acc_rate, grad = dqn.learn_with_one_episode(s=state_action_r, a=action_r, s_=state_action_next_r,
                #                                             isterminal=isterminal_r, r=rewards_r,
                #                                             command_size=command_size_r)
                # print("loss:{},acc_rate:{}".format(loss, acc_rate))

                for _ in range(0,train_round):
                    iters_num = int(replay_buffer.size * 2 / 3 / batch_size)
                    # print("iters_num:{}".format(iters_num))
                    replay_buffer.shuffle()
                    for _ in range(0,iters_num):
                        state_action_r,action_r,state_action_next_r,isterminal_r,rewards_r, command_size_r, command_size_next_r,dead_e_index_r = replay_buffer.get_sample(min(batch_size,replay_buffer.size))
                        loss, acc_rate, grad = dqn.learn_with_one_episode(s=state_action_r,a=action_r,s_=state_action_next_r,isterminal=isterminal_r,r=rewards_r,command_size=command_size_r)

                        nTrain += 1
                        print("nTrain: {}".format(nTrain))
                        print("loss:{},acc_rate:{}".format(loss,acc_rate))
                        wfTo(args.result + '/loss.txt', 'episodes: {} nTrain: {}\nloss: {}\n\n'.format(episodes, nTrain, str(loss)))
                        wfTo(args.result + '/acc_rate.txt', 'episodes: {} nTrain: {}\nacc_rate: {}\n\n'.format(episodes, nTrain, str(acc_rate)))

                        if printWeight and nTrain % 20 == 0:
                            if not os.path.exists(weights_dir + '/learned_' + str(nTrain)):
                                os.mkdir(weights_dir + '/learned_' + str(nTrain))
                            dqn.saveParaTo(weights_dir + '/learned_' + str(nTrain))
                            printGrad(weights_dir + '/learned_' + str(nTrain), grad)

                print("end train!")
            #--------------------------end train---------------------------------
                #print cumulative_reward
                if episodes % CONFIG.episode_to_save == 0 and episodes != 0:
                    dqn.saver.save(dqn.sess, args.result + './model/model.ckpt', global_step=episodes)
