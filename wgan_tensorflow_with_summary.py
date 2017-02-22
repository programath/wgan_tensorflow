import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 32
X_dim = 784
z_dim = 10
h_dim = 128

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histogram', var)

def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

class Weissian_GAN:
    def __init__(self):

        self.graph = tf.Graph()
        self.mb_size = 32
        self.X_dim = 784
        self.z_dim = 10
        self.h_dim = 128

        self.define_graph()
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('out/board/', self.graph)

    def define_graph(self):
        with self.graph.as_default():
            with tf.name_scope('input'):
                self.X = tf.placeholder(tf.float32, shape=[None, X_dim], name = 'input_data')
                self.z = tf.placeholder(tf.float32, shape=[None, z_dim], name = 'random_noise')

            with tf.variable_scope('Discriminator'):
                with tf.variable_scope('layer1'):
                    self.D_W1 = tf.get_variable("weights", initializer=xavier_init([X_dim, h_dim]))
                    variable_summaries(self.D_W1)
                    self.D_b1 = tf.get_variable("biases", [h_dim], initializer=tf.constant_initializer(0.0))
                    variable_summaries(self.D_b1)
                    D_h1 = tf.nn.relu(tf.matmul(self.X, self.D_W1) + self.D_b1)
                    tf.summary.histogram('hidden1', D_h1)
                with tf.variable_scope('layer2'):
                    self.D_W2 = tf.get_variable("weights", initializer=xavier_init([h_dim, 1]))
                    variable_summaries(self.D_W2)
                    self.D_b2 = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0.0))
                    variable_summaries(self.D_b2)
                    D_real = tf.matmul(D_h1, self.D_W2) + self.D_b2
                    tf.summary.histogram('D_real', D_real)

                self.theta_D = [self.D_W1, self.D_W2, self.D_b1, self.D_b2]
                self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.theta_D]

            with tf.variable_scope('Generator'):
                with tf.variable_scope('layer1'):
                    self.G_W1 = tf.get_variable("weights", initializer=xavier_init([z_dim, h_dim]))
                    variable_summaries(self.G_W1)
                    self.G_b1 = tf.get_variable("biases", [h_dim], initializer=tf.constant_initializer(0.0))
                    variable_summaries(self.G_b1)
                    G_h1 = tf.nn.relu(tf.matmul(self.z, self.G_W1) + self.G_b1)
                with tf.variable_scope('layer2'):
                    self.G_W2 = tf.get_variable("weights", initializer=xavier_init([h_dim, X_dim]))
                    variable_summaries(self.G_W2)
                    self.G_b2 = tf.get_variable("biases", [X_dim], initializer=tf.constant_initializer(0.0))
                    variable_summaries(self.G_b2) 
                    G_log_prob = tf.matmul(G_h1, self.G_W2) + self.G_b2
                    G_prob = tf.nn.sigmoid(G_log_prob)               
                self.theta_G = [self.G_W1, self.G_W2, self.G_b1, self.G_b2]

            with tf.variable_scope('Discriminator', reuse = True):
                with tf.variable_scope('layer1'):
                    D_W1 = tf.get_variable("weights", initializer=xavier_init([X_dim, h_dim]))
                    variable_summaries(D_W1)
                    D_b1 = tf.get_variable("biases", [h_dim], initializer=tf.constant_initializer(0.0))
                    variable_summaries(D_b1)
                    DG_h1 = tf.nn.relu(tf.matmul(G_prob, D_W1) + D_b1)

                with tf.variable_scope('layer2'):
                    D_W2 = tf.get_variable("weights", initializer=xavier_init([h_dim, 1]))
                    variable_summaries(D_W2)
                    D_b2 = tf.get_variable("biases", [1], initializer=tf.constant_initializer(0.0))
                    variable_summaries(D_b2)
                    D_fake = tf.matmul(DG_h1, D_W2) + D_b2

            with tf.name_scope('Loss'):
                with tf.name_scope('D_loss'):
                    self.D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
                with tf.name_scope('G_loss'):
                    self.G_loss = -tf.reduce_mean(D_fake)

            with tf.name_scope('optimizer'):
                self.D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                        .minimize(-self.D_loss, var_list=self.theta_D))
                self.G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
                        .minimize(self.G_loss, var_list=self.theta_G)) 

    def run_graph(self):
        def sample_z(m, n):
            return np.random.uniform(-1., 1., size=[m, n])

        with self.sess as sess:
            sess.run(tf.global_variables_initializer())
            i = 0
            for it in range(2000):
                for _ in range(5):
                    X_mb, _ = mnist.train.next_batch(mb_size)

                    _, D_loss_curr, _ = sess.run(
                        [self.D_solver, self.D_loss, self.clip_D],
                        feed_dict={self.X: X_mb, self.z: sample_z(mb_size, z_dim)}
                    )
                    #summary_writer.add_summary(summary, _)

                _, G_loss_curr = sess.run(
                    [self.G_solver, self.G_loss],
                    feed_dict={self.z: sample_z(mb_size, z_dim)}
                )
                
                if it % 100 == 0:
                    print('Iter: {}; D loss: {:.4}; G_loss: {:.4}'
                          .format(it, D_loss_curr, G_loss_curr))
                    

                    # if it % 1000 == 0:
                    #     samples = sess.run(G_sample, feed_dict={z: sample_z(16, z_dim)})

                    #     fig = plot(samples)
                    #     plt.savefig('out/{}.png'
                    #                 .format(str(i).zfill(3)), bbox_inches='tight')
                    #     i += 1
                    #     plt.close(fig)




if __name__ == '__main__':
    if not os.path.exists('out/'):
        os.makedirs('out/')
    wan = Weissian_GAN()
    wan.run_graph()



# saver = tf.train.Saver()
# merged_summary_op = tf.summary.merge_all()
# merged_summary_D = tf.summary.merge([tf.summary.scalar('D_Loss',D_loss)])
# merged_summary_G = tf.summary.merge([tf.summary.scalar('G_loss',G_loss)])
# 






# save_path = saver.save(sess, "out/model.ckpt")
# print "Model saved in file: ", save_path
