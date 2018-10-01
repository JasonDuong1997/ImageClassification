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

def dropout(x, drop_rate=0.5, is_training=True):
	return tf.layers.dropout(x, rate=drop_rate, training=is_training)


### MODEL DEFINITION ###
def Model(x, WIDTH, HEIGHT, n_outputs, is_training):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 8*8

	# DEFINING WEIGHTS
	# Convolution (conv):   [filter_width, filter_height, channels, # of filters]
	# Fully-Connected (fc): [# of neurons in input layer, # of neurons to output]
	# Output (out): 		[# of model outputs]
	W_conv1 = weight([5,5,  3, 8], 	n_inputs=W_conv_input, name="W_conv1")
	W_conv2 = weight([3,3,  8, 16], n_inputs=3*8, name="W_conv2")
	W_conv3 = weight([5,5, 16, 24], n_inputs=8*12, name="W_conv3")
	W_conv4 = weight([3,3, 24, 48], n_inputs=12*16, name="W_conv4")
	W_fc1   = weight([W_fc_input*48, 480], 	n_inputs=21*21, name="W_fc1")
	W_fc2   = weight([480, 120],           	n_inputs=400, name="W_fc2")
	W_fc3   = weight([120, 30],             	n_inputs=35, name="W_fc3")
	W_out   = weight([30, n_outputs],              	n_inputs=16, name="W_fc4")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1 = bias([8],   	name="B_conv1")
	B_conv2 = bias([16],   	name="B_conv2")
	B_conv3 = bias([24],   	name="B_conv3")
	B_conv4 = bias([48],   	name="B_conv4")
	B_fc1   = bias([480], 	name="B_fc1")
	B_fc2   = bias([120],  	name="B_fc2")
	B_fc3   = bias([30],   	name="B_fc3")
	B_out   = bias([n_outputs], name="B_out")

	# DEFINING PilotNetV2 ARCHITECTURE
	# Input Image(width = 80, height = 60, RGB) ->
	# Normalization ->
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu ->
	# Convolution(5x5) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Convolution(3x3) -> Relu ->
	# Fully Connected Layer(1164) -> Relu -> Dropout
	# Fully Connected Layer(100) -> Relu -> Dropout
	# Fully Connected Layer(50) -> Relu ->
	# Output -> Steering Angle
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	# to normalize in hidden layers, add the normalization layer:
	# 1. right after fc or conv layers
	# 2. right before non-linearities
	normalized = tf.layers.batch_normalization(x, training=is_training, trainable=True)
	# normalized = relu(normalized)	# TESTING

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=2)
	conv1 = relu(conv1)

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=1)
	conv2 = relu(conv2)

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides=2)
	conv3 = relu(conv3)

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides=1)
	conv4 = relu(conv4)


	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv4, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	# fc1 = dropout(fc1, 0.2, is_training)

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	# fc2 = dropout(fc2, 0.5, is_training)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)
	# fc3 = dropout(fc3, 0.5, is_training)

	output = tf.matmul(fc3, W_out) + B_out

	return output