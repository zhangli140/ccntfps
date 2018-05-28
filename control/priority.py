import tensorflow as tf
import numpy as np

class PriorityModel(object):
    def __init__(self, s_dim, a_dim, H_units, scope):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.H_units = H_units
        self.s = tf.placeholder(tf.float32, [None, s_dim])
        self.reward = tf.placeholder(tf.float32, [None,])
        self.a = tf.placeholder(tf.float32, [None, a_dim])
        #self.max_gradient = None
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)

        with tf.name_scope(scope):
            with tf.name_scope('hidden_1'):
                h1 = tf.layers.dense(self.s, self.H_units, activation=tf.nn.relu)
                #w1 = tf.Variable(tf.div(tf.random_normal([self.s_dim, self.H_units]),np.sqrt(self.s_dim)))
                #b1 = tf.Variable(tf.constant(0.0, shape=[self.H_units]))
                #h1_raw = tf.nn.relu(tf.matmul(self.s, w1) + b1)
                #mean_pool = tf.reduce_mean(h1_raw, axis=0)
                #max_pool = tf.reduce_max(h1_raw, axis=0)
                #h1 = tf.concat([mean_pool, max_pool], axis=0)
                #h1 = tf.reshape(h1, [1, -1])
            with tf.name_scope('hidden_2'):
                op = tf.layers.dense(h1, self.a_dim)
                #w2 = tf.Variable(tf.div(tf.random_normal([2 * self.H_units, self.a_dim]), np.sqrt(self.H_units)))
                #b2 = tf.Variable(tf.constant(0.0, shape=[ self.a_dim]))
                #print(tf.get_shape(h1))
                #op = tf.matmul(h1, w2) + b2
                ten = tf.constant(10, dtype=tf.float32)
                op = tf.div(op, ten)
                op = op - tf.reduce_max(op)
                self.logp = tf.nn.softmax(op)

                #print(self.logp.get_shape(), self.a.get_shape())
                #self.prob = tf.nn.softmax(self.logp)

            # optimizer
            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=1e-4, decay=0.99)
            # loss
            self.loss = -tf.reduce_mean(
                tf.reduce_sum(self.a * tf.log(self.logp), axis=1))
            #print(self.loss.get_shape())
            # gradient
            self.gradient = self.optimizer.compute_gradients(self.loss)
            # policy gradient
            for i, (grad, var) in enumerate(self.gradient):
                if grad is not None:
                    pg_grad = grad * self.reward
                    # gradient clipping
                    #pg_grad = tf.clip_by_value(
                        #pg_grad, -self.max_gradient, self.max_gradient)
                    self.gradient[i] = (pg_grad, var)
            # train operation (apply gradient)
            self.train_op = self.optimizer.apply_gradients(self.gradient)
            self.saver = tf.train.Saver(max_to_keep=200)
            self.sess.run(tf.global_variables_initializer())

    def calc_priority(self, state):
        state = state.reshape((1, -1))
        pi = self.sess.run(self.logp, feed_dict={self.s: state})[0]
        return pi


    def learn(self, s, a, r):
        feed_dict={}
        feed_dict[self.s] = s.reshape((1, -1))
        feed_dict[self.a] = [a]
        feed_dict[self.reward] = r
        self.sess.run(self.train_op, feed_dict=feed_dict)
        #ls = self.sess.run(self.loss, feed_dict=feed_dict)
        #lp = self.sess.run(self.logp, feed_dict=feed_dict)
        #gt = self.sess.run(self.gradient, feed_dict=feed_dict)
        #print(ls, lp, a, gt)


