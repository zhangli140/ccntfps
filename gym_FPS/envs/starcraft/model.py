# -*-coding:utf-8 -*-

import tensorflow as tf
import numpy as np
from .Config import *
from .memory import *
from tensorflow.contrib import rnn

CONFIG = Config()

LR_A = CONFIG.lr_a  # learning rate for actor
LR_C = CONFIG.lr_c  # learning rate for critic
GAMMA = CONFIG.gamma  # reward discount
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
MEMORY_CAPACITY = CONFIG.memory_capacity
BATCH_SIZE = CONFIG.batch_size
REPLAY_START_SIZE=CONFIG.replay_start_size


class DDPG(object):
    def __init__(self,sess, a_dim, s_dim, a_bound, actor, critic):
        self.memory = np.zeros((MEMORY_CAPACITY), dtype=object)
        self.pointer = 0
        self.sess = sess
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S1 = tf.placeholder(tf.float32,[None,s_dim],'s1')
        self.global_step = tf.Variable(0, trainable=False)
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.S1_ = tf.placeholder(tf.float32,[None,s_dim],'S1_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.unit_size = tf.placeholder(tf.int32, name='unit_size')
        self.unit_size_ = tf.placeholder(tf.int32, name='unit_size_')
        self.variable_summaries('reward', self.R)
        with tf.variable_scope(actor):
            self.a = self._build_a(self.S, self.S1,scope='eval', trainable=True, unit_size=self.unit_size)
            self.variable_summaries('eval_a', self.a)
            a_ = self._build_a(self.S_, self.S1_,scope='target', trainable=False, unit_size=self.unit_size_)
            self.variable_summaries('target_a', a_)
        with tf.variable_scope(critic):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True, unit_size=self.unit_size)
            self.variable_summaries('eval_q', q)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False, unit_size=self.unit_size_)
            self.variable_summaries('target_q', q_)
        # networks parameters


        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor + '/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=actor + '/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=critic + '/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=critic + '/target')

        for var in self.ae_params:
            self.variable_summaries(var.op.name, var)
        for var in self.at_params:
            self.variable_summaries(var.op.name, var)
        for var in self.ce_params:
            self.variable_summaries(var.op.name, var)
        for var in self.ct_params:
            self.variable_summaries(var.op.name, var)

        q_target = self.R + GAMMA * q_
        self.variable_summaries('q_target', q_target)
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params,
                                                            global_step=self.global_step)
        a_loss = - tf.reduce_sum(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params,
                                                            global_step=self.global_step)
        tf.summary.scalar('a_loss', a_loss)
        tf.summary.scalar('c_loss', td_error)

        self.merged_summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=CONFIG.max_to_keep)
#        self.summary_writer = tf.summary.FileWriter(CONFIG.summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def variable_summaries(self, name, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '_mean', mean)
            with tf.name_scope(name + '_stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar(name + '_stddev', stddev)
            tf.summary.scalar(name + '_max', tf.reduce_max(var))
            tf.summary.scalar(name + '_min', tf.reduce_min(var))
            tf.summary.histogram(name + '_histogram', var)

    def choose_action(self, s, s1, unit_size):
        a_out = self.sess.run(self.a, {self.S: s,self.S1:s1, self.unit_size: unit_size})

        return a_out

    def learn(self):
        # hard replace parameters
        if self.a_replace_counter % REPLACE_ITER_A == 0:#actor target networks更新
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % REPLACE_ITER_C == 0:#critic target network更新
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter += 1
        self.c_replace_counter += 1
        if self.pointer>=MEMORY_CAPACITY:
            indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        elif self.pointer>=REPLAY_START_SIZE:
            indices = np.random.choice(self.pointer,size=BATCH_SIZE)#改为从start开始学
        else:
            print ("replay_start_size is smaller than batch_size!")
            return
        bt = self.memory[indices]
        for i in range(bt.size): #原来是这个for循环大大增加了运行时间。
            self.sess.run([self.atrain], {self.S: bt[i].s,self.S1:bt[i].s1,
                                        self.unit_size: bt[i].unit_size})
            reward = np.array(bt[i].r).reshape((-1, 1))#这里张煜reshape了一下。其实不仅reward要reshape，state最好也要reshape
            summary,_ = self.sess.run([self.merged_summary_op,self.ctrain], {self.S: bt[i].s, self.S1:bt[i].s1,
                                                                               self.a: bt[i].a,
                                                                               self.R: reward, self.S_: bt[i].s_,
                                                                               self.S1_:bt[i].s1_,
                                                                               self.unit_size: bt[i].unit_size,
                                                                               self.unit_size_:bt[i].unit_size_})
#            self.summary_writer.add_summary(summary, self.sess.run(self.global_step))

    def store_transition(self, s, s1, a, r, s_, s1_, unit_size, unit_size_):
        mymemory = Memory(s,s1, a, r, s_,s1_, unit_size,unit_size_)
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index] = mymemory
        self.pointer += 1

    def _build_a(self, s, s1, scope, trainable, unit_size):
        with tf.variable_scope(scope):  # s:[enemy_size+unit_size,hidden_size]
            w1 = tf.Variable(
                tf.truncated_normal([self.s_dim, CONFIG.hidden_size * 2], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)  # 2*CONFIG.hidden_size
            b1 = tf.Variable(tf.zeros([CONFIG.hidden_size * 2, ]))
            x = tf.nn.elu(tf.matmul(s, w1) + b1)  # [unit_size+enemy_size,2*hidden_size]
            x1 = tf.nn.elu(tf.matmul(s1, w1) + b1)
            mean_pool = tf.reduce_mean(x, axis=0)
            max_pool = tf.reduce_max(x, axis=0)
            feature = tf.concat([ mean_pool, max_pool], axis=0)
            feature = tf.reshape(feature, [1, 4 * CONFIG.hidden_size])
            feature = tf.multiply(feature, tf.zeros([unit_size, 1]) + 1)  # unit_size,4*config.hidden_size
            feature = tf.concat([x1,feature], axis = 1)
            final_feature = tf.reshape(feature, [1, unit_size, 6 * CONFIG.hidden_size])

            cell = tf.contrib.rnn.BasicLSTMCell(num_units=CONFIG.hidden_size)
#            cell = tf.contrib.rnn.BasicRNNCell(num_units=CONFIG.hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=final_feature,
                                                              dtype=tf.float32, scope=scope)
            outputs_all = tf.concat(outputs, 1)
            outputs_all = tf.reshape(outputs_all, [unit_size, 2 * CONFIG.hidden_size])
            w2 = tf.Variable(
                tf.truncated_normal([2 * CONFIG.hidden_size, self.a_dim], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)
            b2 = tf.Variable(tf.zeros([self.a_dim, ], dtype=tf.float32), trainable=trainable)
            action_value = tf.nn.tanh(tf.matmul(outputs_all, w2) + b2)
            return action_value

    def _build_c(self, s, a, scope, trainable, unit_size):
        with tf.variable_scope(scope):
            w1 = tf.Variable(
                tf.truncated_normal([self.s_dim, CONFIG.hidden_size * 2], stddev=CONFIG.std, dtype=tf.float32),
                trainable=trainable)  # 10*CONFIG.hidden_size
            b1 = tf.Variable(tf.zeros([CONFIG.hidden_size * 2, ]))
            x = tf.nn.elu(tf.matmul(s, w1) + b1)  # [unit_size+enemy_size,10*hidden_size]
            mean_pool = tf.reduce_mean(x, axis=0)
            max_pool = tf.reduce_max(x, axis=0)
            feature = tf.concat([mean_pool, max_pool],axis=0)
            feature = tf.reshape(feature, [1, 4 * CONFIG.hidden_size])
            feature = tf.multiply(feature, tf.zeros([unit_size, 1]) + 1)  # unit_size,4*config.hidden_size
            action_and_q = tf.concat([a, feature], axis=1)  # unit_size,4*config.hidden_size+3
            input_feature = tf.reshape(action_and_q, [1, unit_size, CONFIG.hidden_size * 4 + 3])
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=CONFIG.hidden_size)
#            cell = tf.contrib.rnn.BasicRNNCell(num_units=CONFIG.hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell, cell_bw=cell, inputs=input_feature,
                                                              dtype=tf.float32, scope=scope)
            outputs = tf.concat(outputs, 1)
            outputs = tf.reshape(outputs, [unit_size, 2 * CONFIG.hidden_size])
            w2 = tf.Variable(tf.truncated_normal([2 * CONFIG.hidden_size, 1], stddev=CONFIG.std, dtype=tf.float32),
                             trainable=trainable)
            b2 = tf.Variable(tf.zeros([1, ], dtype=tf.float32), trainable=trainable)
            Q_value = tf.matmul(outputs, w2) + b2
            return Q_value  # Q(s,a)
