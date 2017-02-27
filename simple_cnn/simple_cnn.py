import tensorflow as tf
from tensorflow.examples.tutorials.mnist import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

image_size = 28
image_channels = 1
conv1_Channels = image_channels
conv2_Channels = 32
BATCH_SIZE = 64
SEED = 66478
num_labels = 10

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
train_size = mnist.train.num_examples


def error_rate(prediction, labels):
	return 100 - (100 * np.sum(np.argmax(prediction, 1)==labels) / prediction.shape[0])

class CNN_network:
	def __init__(self):


		self.graph = tf.Graph()
		self.define_graph()
		self.sess = tf.Session(graph=self.graph)
		self.writer = tf.summary.FileWriter('out/cnn/', self.graph)
		

	def define_graph(self):
		with self.graph.as_default():
			with tf.name_scope("input"):
				self.train_data_nodes = tf.placeholder(
					tf.float32, shape=[BATCH_SIZE, image_size, image_size, image_channels], name = 'input_data')
				self.train_labels_nodes = tf.placeholder(
					tf.int64, shape=[BATCH_SIZE,], name = 'input_label')
			with tf.variable_scope("conv1"):
				conv1_weights = tf.get_variable("weights", [5, 5, conv1_Channels, 32], initializer=tf.random_normal_initializer(stddev=0.1))
				conv1_biases = tf.get_variable("biases", [32], initializer=tf.constant_initializer(0.0))
				conv = tf.nn.conv2d(self.train_data_nodes, conv1_weights, strides=[1,1,1,1], padding = 'SAME')
				relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
				pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
				
			with tf.variable_scope("conv2"):
				conv2_weights = tf.get_variable("weights", [5, 5, conv2_Channels, 64], initializer=tf.random_normal_initializer(stddev=0.1))
				conv2_biases = tf.get_variable("biases", [64], initializer=tf.constant_initializer(0.1))
				conv = tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding = 'SAME')
				relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
				pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			pool_shape = pool.get_shape().as_list()
			reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])


			with tf.variable_scope("fc1"):
				fc1_weights = tf.get_variable("weights", [image_size // 4 * image_size // 4 * 64, 512], initializer=tf.random_normal_initializer(stddev=0.1))
				fc1_biases = tf.get_variable("biases", [512], initializer=tf.constant_initializer(0.1))
				hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
				hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
			with tf.variable_scope("fc2"):
				fc2_weights = tf.get_variable("weights", [512, num_labels], initializer=tf.random_normal_initializer(stddev=0.1))
				fc2_biases = tf.get_variable("biases", [num_labels], initializer=tf.constant_initializer(0.1))
				out = tf.matmul(hidden, fc2_weights) + fc2_biases
			with tf.name_scope("loss"):
				regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + 
								tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
				self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(out, self.train_labels_nodes))

				self.loss += 5e-4 * regularizers
			batch = tf.Variable(0, tf.float32, name='batch')
			self.learning_rate = tf.train.exponential_decay(
					0.01,
					batch * BATCH_SIZE,
					train_size,
					0.95,
					staircase=True)
			self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9).minimize(self.loss, global_step=batch)

			self.train_prediction = tf.nn.softmax(out)

			tf.summary.scalar('loss', self.loss)	
			tf.summary.scalar('learning_rate', self.learning_rate)
			tf.summary.histogram('conv1_weights', conv1_weights)
			tf.summary.histogram('conv2_weights', conv2_weights)
			tf.summary.histogram('fc1_weights', fc1_weights)
			tf.summary.histogram('fc2_weights', fc2_weights)
			self.merge = tf.summary.merge_all()


	def run_graph(self):
		with self.sess as sess:
			sess.run(tf.global_variables_initializer())

			for step in range(10 * train_size // BATCH_SIZE):
				batch_data, batch_labels = mnist.train.next_batch(BATCH_SIZE)
				batch_data = batch_data.reshape(BATCH_SIZE, image_size, image_size, image_channels)
				feed_dict = {self.train_data_nodes: batch_data,
							 self.train_labels_nodes: batch_labels}
				sess.run(self.optimizer, feed_dict=feed_dict)

				if step % 100 == 0:
					merge, l, lr, p = sess.run([self.merge, self.loss, self.learning_rate, self.train_prediction], feed_dict=feed_dict)
					self.writer.add_summary(merge, step)
					print('Step %d (epoch %.2f) ' %
						   (step, float(step) * BATCH_SIZE / train_size))
					print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
					print('Minibatch error rate: %.3f%%' % error_rate(p, batch_labels))
					#print p

if __name__ == '__main__':
	if not os.path.exists('out/'):
		os.makedirs('out/')
	simple_cnn = CNN_network()
	simple_cnn.run_graph()











