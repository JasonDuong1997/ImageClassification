import tensorflow as tf
import math


### TF VARIABLES ###
def weight(shape, n_inputs, name):
	# calculating standard deviation based on number of inputs
	std_dev = math.sqrt(2.0/n_inputs)
	# numbers chosen more than 2 std devs away are thrown away and re-picked
	initial_val = tf.truncated_normal(shape, stddev=std_dev)
	return tf.Variable(initial_val, name=name)

def bias(shape, name):
	initial_val = tf.constant(0.1, shape=shape)
	return tf.Variable(initial_val, name=name)


### ACTIVATION/TRANSFER FUNCTIONS ###
def relu(x):
	return tf.nn.relu(x)

def tanh(x):
	return tf.tanh(x)

def sigmoid(x):
	return tf.sigmoid(x)


### OPERATIONS ###
def conv2d(x, weight, bias, strides=1):
	return tf.nn.conv2d(x, weight, strides=[1, strides,strides, 1], padding="SAME") + bias

def max_pool2d(x, strides=2):
	return tf.nn.max_pool(x, ksize=[1, strides,strides, 1], strides=[1, strides,strides, 1], padding="SAME")

def normalize(x, is_training):
	return tf.layers.batch_normalization(x, training=is_training, trainable=True)


def dropout(x, drop_rate=0.5, is_training=True):
	return tf.layers.dropout(x, rate=drop_rate, training=is_training)


### MODEL DEFINITION ###
def Model(x, WIDTH, HEIGHT, n_outputs, is_training):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 8*8

	# DEFINING WEIGHTS
	# Convolution (W_conv#):   	[filter_width, filter_height, channels, # of filters]
	# Fully-Connected (W_fc#): 	[# of neurons in input layer, # of neurons to output]
	# Output (W_out): 		 	[# of model outputs]
	W_conv1 = weight([5,5,   3, 32],  n_inputs=W_conv_input, name="W_conv1")
	W_conv2 = weight([3,3,  32, 48],  n_inputs=32*48, 		 name="W_conv2")
	W_conv3 = weight([5,5,  48, 64],  n_inputs=48*96, 		 name="W_conv3")
	W_conv4 = weight([3,3,  64, 128], n_inputs=96*192, 		 name="W_conv4")
	W_conv5 = weight([3,3, 128, 256], n_inputs=192*256, 	 name="W_conv5")
	W_conv6 = weight([3,3, 256, 128], n_inputs=256*192, 	 name="W_conv6")
	W_conv7 = weight([3,3, 128, 128], n_inputs=256*192, 	 name="W_conv7")
	W_fc1   = weight([W_fc_input*128, 1024], n_inputs=800, 	 name="W_fc1")
	W_fc2   = weight([1024, 256],            n_inputs=256, 	 name="W_fc2")
	W_fc3   = weight([256, 128],             n_inputs=128,   name="W_fc3")
	W_out   = weight([128, n_outputs],       n_inputs=10,    name="W_out")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1 = bias([32],   		name="B_conv1")
	B_conv2 = bias([48],   		name="B_conv2")
	B_conv3 = bias([64],   		name="B_conv3")
	B_conv4 = bias([128],   	name="B_conv4")
	B_conv5 = bias([256],   	name="B_conv5")
	B_conv6 = bias([128],   	name="B_conv6")
	B_conv7 = bias([128],   	name="B_conv7")
	B_fc1   = bias([1024], 		name="B_fc1")
	B_fc2   = bias([256],  		name="B_fc2")
	B_fc3   = bias([128],   	name="B_fc3")
	B_out   = bias([n_outputs], name="B_out")

	# DEFINING PilotNetV2 ARCHITECTURE
	# Input Image(width = 32, height = 32, RGB) ->
	# Normalization ->
	# Convolution(5x5) -> Relu -> Normalization ->
	# Convolution(3x3) -> Relu -> Normalization ->
	# Convolution(5x5) -> Relu -> Normalization ->
	# Convolution(3x3) -> Relu -> Normalization ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(1024) -> Relu -> Dropout
	# Fully Connected Layer(256)  -> Relu -> Dropout
	# Fully Connected Layer(128)  -> Relu -> Dropout
	# Output -> Softmax -> Classification
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	# to normalize in hidden layers, add the normalization layer:
	# 1. right after fc or conv layers
	# 2. right before non-linearities
	normalized = normalize(x, is_training=is_training)

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=1)
	conv1 = relu(conv1)
	conv1 = max_pool2d(conv1, strides=2)

	conv2 = normalize(conv1, is_training=is_training)
	conv2 = conv2d(conv2, W_conv2, B_conv2, strides=1)
	conv2 = relu(conv2)

	conv3 = normalize(conv2, is_training=is_training)
	conv3 = conv2d(conv3, W_conv3, B_conv3, strides=1)
	conv3 = relu(conv3)

	conv4 = normalize(conv3, is_training=is_training)
	conv4 = conv2d(conv4, W_conv4, B_conv4, strides=1)
	conv4 = relu(conv4)

	conv5 = normalize(conv4, is_training=is_training)
	conv5 = conv2d(conv5, W_conv5, B_conv5, strides=1)
	conv5 = relu(conv5)
	conv5 = max_pool2d(conv5, strides=2);

	conv6 = conv2d(conv5, W_conv6, B_conv6, strides=1)
	conv6 = relu(conv6)

	conv7 = conv2d(conv6, W_conv7, B_conv7, strides=1)
	conv7 = relu(conv7)

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv7, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	fc1 = dropout(fc1, 0.5, is_training)

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.5, is_training)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)
	fc3 = dropout(fc3, 0.5, is_training)

	output = tf.matmul(fc3, W_out) + B_out	# output activation is SoftMax, but is handled in training

	return output
