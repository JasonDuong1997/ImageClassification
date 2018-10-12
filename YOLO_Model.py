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
def activation(x, type):
	if (type == "relu"):
		return tf.nn.relu(x)
	elif (type == "leaky_relu"):
		return 	return tf.nn.leaky_relu(x, alpha=0.1)
	elif (type == "siggmoid"):
		return tf.sigmoid(x)
	elif (type == "tanh"):
		return tf.tanh(x)
	else:
		print("ERROR: No activation type specified.")
		return -1

def relu(x):
	return tf.nn.relu(x)

def leaky_relu(x):
	return tf.nn.leaky_relu(x, alpha=0.1)

def tanh(x):
	return tf.tanh(x)

def sigmoid(x):
	return tf.sigmoid(x)


### OPERATIONS ###
def conv2d(x, weight, bias, strides=1):
	return tf.nn.conv2d(x, weight, strides=[1, strides,strides, 1], padding="SAME") + bias

def max_pool2d(x, kernel_size, strides):
	return tf.nn.max_pool(x, ksize=[1, strides,strides, 1], strides=[1, strides,strides, 1], padding="SAME")

def dropout(x, drop_rate=0.5, is_training=True):
	return tf.layers.dropout(x, rate=drop_rate, training=is_training)


### MODEL DEFINITION ###
def YOLO_Model(x, WIDTH, HEIGHT, n_outputs, is_training):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 8*10

	# DEFINING WEIGHTS
	# Convolution (conv):   [filter_width, filter_height, channels, # of filters]
	# Fully-Connected (fc): [# of neurons in input layer, # of neurons to output]
	# Output (out): 		[# of model outputs]
	W_conv1  = weight([7,7,    3, 64], 	 n_inputs=W_conv_input, name="W_conv1")
	W_conv2  = weight([3,3,   64, 192],  n_inputs=3*8,   name="W_conv2")
	W_conv3  = weight([1,1,  192, 128],  n_inputs=8*12,  name="W_conv3")
	W_conv4  = weight([3,3,  128, 256],  n_inputs=12*16, name="W_conv4")
	W_conv5  = weight([1,1,  256, 256],  n_inputs=16*21, name="W_conv5")
	W_conv6  = weight([3,3,  256, 512],  n_inputs=16*21, name="W_conv6")
	W_conv7  = weight([1,1,  512, 256],  n_inputs=16*21, name="W_conv7")
	W_conv8  = weight([3,3,  256, 512],  n_inputs=16*21, name="W_conv8")
	W_conv9  = weight([1,1,  512, 512],  n_inputs=16*21, name="W_conv9")
	W_conv10 = weight([3,3,  512, 1024], n_inputs=16*21, name="W_conv10")
	W_conv11 = weight([1,1, 1024, 512],  n_inputs=16*21, name="W_conv11")
	W_conv12 = weight([3,3,  512, 1024], n_inputs=16*21, name="W_conv12")
	W_conv13 = weight([3,3, 1024, 1024], n_inputs=16*21, name="W_conv13")
	W_conv14 = weight([3,3, 1024, 1024], n_inputs=16*21, name="W_conv14")
	W_conv15 = weight([3,3, 1024, 1024], n_inputs=16*21, name="W_conv15")
	W_conv16 = weight([3,3, 1024, 1024], n_inputs=16*21, name="W_conv16")
	W_fc1    = weight([W_fc_input*1024, 4096],	n_inputs=21*21, name="W_fc1")
	W_fc2    = weight([4096, 30],              	n_inputs=400, name="W_fc2")
	W_out    = weight([12, n_outputs],       	n_inputs=5, name="W_out")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1  = bias([64],   name="B_conv1")
	B_conv2  = bias([192],  name="B_conv2")
	B_conv3  = bias([128],  name="B_conv3")
	B_conv4  = bias([256],  name="B_conv4")
	B_conv5  = bias([256],  name="B_conv5")
	B_conv6  = bias([512], 	name="B_fc1")
	B_conv7  = bias([256],  name="B_fc2")
	B_conv8  = bias([512],  name="B_fc3")
	B_conv9  = bias([512],  name="B_fc4")
	B_conv10 = bias([1024], name="B_fc4")
	B_conv11 = bias([512],  name="B_fc4")
	B_conv12 = bias([1024], name="B_fc4")
	B_conv13 = bias([1024], name="B_fc4")
	B_conv14 = bias([1024], name="B_fc4")
	B_conv15 = bias([1024], name="B_fc4")
	B_conv16 = bias([1024], name="B_fc4")
	B_conv17 = bias([1024], name="B_fc4")
	B_fc1 = bias([4096], 	name="B_fc1")
	B_out   = bias([n_outputs], name="B_out")

	# DEFINING PilotNetV2 ARCHITECTURE
	# Input Image(width = ?, height = ?, RGB) ->
	# Convolution(7x7) -> Relu -> MaxPool ->
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

	activation_type = "leaky_relu"

	# Convolution Group 1
	conv1 = conv2d(x, W_conv1, B_conv1, strides=2)
	conv1 = activation(conv1, activation_type)
	maxP1 = max_pool2d(conv1, 2, strides=2)

	# Convolution Group 2
	conv2 = conv2d(maxP1, W_conv2, B_conv2, strides=1)
	conv2 = activation(conv2, activation_type)
	maxP2   = max_pool2d(conv2, 2, strides=2)

	# Convolution Group 3
	conv3 = conv2d(maxP2, W_conv3, B_conv3, strides=1)
	conv3 = activation(conv3, activation_type)
	conv4 = conv2d(conv3, W_conv4, B_conv4, strides=1)
	conv4 = activation(conv4, activation_type)
	conv5 = conv2d(conv4, W_conv5, B_conv5, strides=1)
	conv5 = activation(conv5, activation_type)
	conv6 = conv2d(conv5, W_conv6, B_conv6, strides=1)
	conv6 = activation(conv6, activation_type)
	maxP3 = max_pool2d(conv6, 2, strides=2)

	# Convolution Group 4
	conv7 = conv2d(maxP3, W_conv7, B_conv7, strides=1)
	conv7 = activation(conv7, activation_type)
	conv8 = conv2d(conv7, W_conv8, B_conv8, strides=1)
	conv8 = activation(conv8, activation_type)
	conv9 = conv2d(conv8, W_conv9, B_conv9, strides=1)
	conv9 = activation(conv9, activation_type)
	conv10 = conv2d(conv9, W_conv10, B_conv10, strides=1)
	conv10 = activation(conv10, activation_type)
	maxP4 = max_pool2d(conv10, 2, strides=2)

	# Convolution Group 5
	conv11 = conv2d(maxP4, W_conv11, B_conv11, strides=1)
	conv11 = activation(conv11, activation_type)
	conv12 = conv2d(conv11, W_conv12, B_conv12, strides=1)
	conv12 = activation(conv12, activation_type)
	conv13 = conv2d(conv12, W_conv13, B_conv13, strides=1)
	conv13 = activation(conv13, activation_type)
	conv14 = conv2d(conv13, W_conv14, B_conv14, strides=2)
	conv14 = activation(conv14, activation_type)

	# flatten to 1 dimension for fully connected layers
	flat_img = tf.reshape(conv5, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = relu(tf.matmul(flat_img, W_fc1) + B_fc1)
	# fc1 = dropout(fc1, 0.2, is_training)

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	# fc2 = dropout(fc2, 0.5, is_training)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)
	# fc3 = dropout(fc3, 0.5, is_training)

	fc4 = relu(tf.matmul(fc3, W_fc4) + B_fc4)

	# pure linear output
	output = tf.matmul(fc4, W_out) + B_out

	return output
