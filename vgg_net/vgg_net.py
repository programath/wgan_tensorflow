import scipy.io
import scipy.misc
import tensorflow as tf
import numpy as np

image_height = 80
image_width = 101
image_channel = 3

ITERATION = 2000
ALPHA = 1
BETA = 10000

vgg_model = 'imagenet-vgg-verydeep-19.mat'
mean_values = np.array([123, 116, 103]).reshape((1,1,1,3))
vgg = scipy.io.loadmat(vgg_model)
vgg_layers = vgg['layers']
content_path = 'small_Starry_Night.jpg'
content_image = None
style_path = 'youhua1.jpg'
style_image = None



def initialize_weights(layer_id):
	weights = vgg_layers[0][layer_id][0][0][2][0][0]
	return tf.constant(weights, name="weights")

def initialize_biases(layer_id):
	biases = vgg_layers[0][layer_id][0][0][2][0][1]
	biases = np.reshape(biases, (biases.size))
	return tf.constant(biases, name="biases")

def initialize_pool(layer_id):
	return vgg_layers[0][layer_id][0][0][3][0], vgg_layers[0][4][0][0][4][0]

def load_image(image_path):
	image = scipy.misc.imread(image_path)
	image = np.reshape(image, ((1,) + image.shape))
	image = image - mean_values
	return image

def save_image(image_path, image):
	image = image + mean_values
	image = image[0]
	image = np.clip(image, 0, 255).astype('uint8')
	scipy.misc.imsave(image_path, image)

def generate_noise_image(content_image, ratio=0.6):
	noise = np.random.uniform(-20,20,(1, image_height, image_width, image_channel)).astype('float32')
	return noise * ratio + content_image.astype('float32') * (1 - ratio)


def conv_relu(input, id):
	weights = initialize_weights(id)
	biases = initialize_biases(id)
	conv = tf.nn.conv2d(input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
	return tf.nn.relu(conv + biases)

def avg_pool(input, id):
	ksize, shape = initialize_pool(id)
	return tf.nn.avg_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


class vgg_net:#vgg-19 conv-net

	def __init__(self):

		self.graph = tf.Graph()
		self.sess = tf.Session(graph = self.graph)
		self.define_graph()
		self.writer = tf.summary.FileWriter('out/vgg_net/', self.graph)

	def define_graph(self):
		with self.graph.as_default():
			with tf.variable_scope("input_data"):
				self.X = tf.get_variable("input_data", shape=(1, image_height, image_width, image_channel), dtype='float32', initializer=tf.constant_initializer(0.0))

			with tf.name_scope("conv1_1"):#0,1,
				conv1_1 = conv_relu(self.X, 0)
			with tf.name_scope("conv1_2"):
				conv1_2 = conv_relu(conv1_1, 2)#2,3
			with tf.name_scope("pool1"):
				pool1 = avg_pool(conv1_2, 4)#4

			with tf.name_scope("conv2_1"):
				conv2_1 = conv_relu(pool1, 5)#5,6
			with tf.name_scope("conv2_2"):
				conv2_2 = conv_relu(conv2_1, 7)#7,8
			with tf.name_scope("pool2"):
				pool2 = avg_pool(conv2_2, 9)

			with tf.name_scope("conv3_1"):
				conv3_1 = conv_relu(pool2, 10)#10 11
			with tf.name_scope("conv3_2"):
				conv3_2 = conv_relu(conv3_1, 12)
			with tf.name_scope("conv3_3"):
				conv3_3 = conv_relu(conv3_2, 14)
			with tf.name_scope("conv3_4"):
				conv3_4 = conv_relu(conv3_3, 16)
			with tf.name_scope("pool3"):
				pool3 = avg_pool(conv3_4, 18)

			with tf.name_scope("conv4_1"):
				conv4_1 = conv_relu(pool3, 19)
			with tf.name_scope("conv4_2"):
				conv4_2 = conv_relu(conv4_1, 21)
			with tf.name_scope("conv4_3"):
				conv4_3 = conv_relu(conv4_2, 23)
			with tf.name_scope("conv4_4"):
				conv4_4 = conv_relu(conv4_3, 25)
			with tf.name_scope("pool4"):
				pool4 = avg_pool(conv4_4, 27)

			with tf.name_scope("conv5_1"):
				conv5_1 = conv_relu(pool4, 28)
			with tf.name_scope("conv5_2"):
				conv5_2 = conv_relu(conv5_1, 30)
			with tf.name_scope("conv5_3"):
				conv5_3 = conv_relu(conv5_2, 32)
			with tf.name_scope("conv5_4"):
				conv5_4 = conv_relu(conv5_3, 34)
			with tf.name_scope("pool4"):
				pool5 = avg_pool(conv5_4, 36)

			def content_loss(content_image):
				self.sess.run(self.X.assign(content_image))					
				x = self.sess.run(conv4_2)
				Nl = tf.shape(conv4_2)[3].eval(session=self.sess)# the number of filters is equal to the size of biases
				Ml = tf.shape(conv4_2)[1].eval(session=self.sess) * tf.shape(conv4_2)[2].eval(session=self.sess) #[batch size, height, width, depth=channel]
				Nl2 = (Nl**2).astype('float32')
				Ml2 = (Ml**2).astype('float32')
				return 0.5 * tf.reduce_sum(tf.pow(x - conv4_2, 2))
			
			def style_loss(style_image):
				self.sess.run(self.X.assign(style_image))	
				STYLE_LAYERS = [
    				(conv1_1, 0.2),#0.5
    				(conv2_1, 0.2),#1.0
    				(conv3_1, 0.2),#1.5
    				(conv4_1, 0.2),#3.0
    				(conv5_1, 0.2),#4.0
				]
				def _style_loss(a, x):
					Nl = tf.shape(a)[3].eval(session=self.sess)
					Ml = tf.shape(a)[1].eval(session=self.sess) * tf.shape(a)[2].eval(session=self.sess)
					a = tf.reshape(a, (Nl, Ml))
					A = tf.matmul(tf.transpose(a),a)
					x = tf.reshape(x, (Nl, Ml))
					G = tf.matmul(tf.transpose(x),x)
					Nl2 = (Nl**2).astype('float32')
					Ml2 = (Ml**2).astype('float32')
					return (1.0/(4 * Nl2 * Ml2)) * tf.reduce_sum(tf.pow(G - A, 2))

				E = [_style_loss(self.sess.run(layer), layer) for layer, _ in STYLE_LAYERS ]
				W = [w for _, w in STYLE_LAYERS]
				return sum([E[l]*W[l] for l in range(len(STYLE_LAYERS))])



			content_loss = content_loss(content_image)
			style_loss = style_loss(style_image)
			total_loss = ALPHA * content_loss + BETA * style_loss

			self.optimizer = tf.train.AdamOptimizer(2.0).minimize(total_loss)

	def run_graph(self):
		with self.sess as sess:
			sess.run(tf.global_variables_initializer())
			input_image = generate_noise_image(content_image)
			sess.run(self.X.assign(input_image))
			print("start training")
			for i in range(0, ITERATION):
				sess.run(self.optimizer)
				print("iter %d is done" % i)
				if i % 100 == 0:
					image = sess.run(self.X)
					image_path = 'out/vgg_net/%d.png' % i 
					save_image(image_path, image)


if __name__ == '__main__':
	content_image = load_image(content_path)
	style_image = load_image(style_path)
	vgg_model = vgg_net()
	vgg_model.run_graph()













	


