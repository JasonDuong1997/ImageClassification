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

# Statistics at 100 Epochs
# 	Trial		Train_Loss		Test_Loss	Train_Acc		Val_Acc
#	1.			94.5			77.9		0.899			0.687
#	2.			67.86			94.1		0.952			0.70
#	3.			120				85.0		0.887			0.702
#	4.			58.0			87.6		0.96			0.743
### MODEL DEFINITION ###
def Model(x, WIDTH, HEIGHT, n_outputs, is_training):
	W_conv_input = WIDTH*HEIGHT*3
	W_fc_input = 4*4

	# DEFINING WEIGHTS
	# Convolution (W_conv#):   	[filter_width, filter_height, channels, # of filters]
	# Fully-Connected (W_fc#): 	[# of neurons in input layer, # of neurons to output]
	# Output (W_out): 		 	[# of model outputs]
	W_conv1 = weight([5,5,    3, 32],  n_inputs=W_conv_input,	name="W_conv1")
	W_conv2 = weight([5,5,   32, 48],  n_inputs=32*48, 		 	name="W_conv2")
	W_conv3 = weight([5,5,   48, 96],  n_inputs=48*96, 		 	name="W_conv3")
	W_conv4 = weight([5,5,   96, 48],  n_inputs=96*48, 		 	name="W_conv4")
	W_conv5 = weight([5,5,   48, 96],  n_inputs=48*96, 		 	name="W_conv5")
	W_conv6 = weight([3,3,   96, 128], n_inputs=96*128, 		name="W_conv6")
	W_conv7 = weight([3,3,  128, 256], n_inputs=128*256, 	 	name="W_conv7")
	W_conv8 = weight([3,3,  256, 192], n_inputs=256*192, 	 	name="W_conv8")
	W_conv9 = weight([3,3,  192, 256], n_inputs=192*256, 	 	name="W_conv9")
	W_conv10 = weight([3,3, 256, 128], n_inputs=256*192, 		name="W_conv10")
	W_conv11 = weight([2,2, 128, 192], n_inputs=256*192, 	 	name="W_conv11")
	W_conv12 = weight([2,2, 192, 256], n_inputs=256*192, 	 	name="W_conv12")
	W_conv13 = weight([2,2, 256, 128], n_inputs=256*192, 	 	name="W_conv13")
	W_conv14 = weight([2,2, 128, 128], n_inputs=256*192, 	 	name="W_conv14")
	W_fc1   = weight([W_fc_input*128, 1024], n_inputs=1024,  	name="W_fc1")
	W_fc2   = weight([1024, 512],            n_inputs=512, 	 	name="W_fc2")
	W_fc3   = weight([512, 256],             n_inputs=256, 	 	name="W_fc3")
	W_fc4   = weight([256, 128],             n_inputs=128, 	 	name="W_fc4")
	W_out   = weight([128, n_outputs],       n_inputs=10,    	name="W_out")
	# DEFINING BIASES
	# Convolution (conv): 	[# number of filters]
	# Fully-Connected (fc): [# number of filters]
	# Output (out): 		[# of outputs]
	B_conv1  = bias([32],   	name="B_conv1")
	B_conv2  = bias([48],   	name="B_conv2")
	B_conv3  = bias([96],   	name="B_conv2a")
	B_conv4  = bias([48],   	name="B_conv2b")
	B_conv5  = bias([96],   	name="B_conv3")
	B_conv6  = bias([128],   	name="B_conv4")
	B_conv7  = bias([256],   	name="B_conv5")
	B_conv8  = bias([192],   	name="B_conv5a")
	B_conv9  = bias([256],   	name="B_conv5b")
	B_conv10 = bias([128],   	name="B_conv6")
	B_conv11 = bias([192],   	name="B_conv7")
	B_conv12 = bias([256],   	name="B_conv8")
	B_conv13 = bias([128],   	name="B_conv9")
	B_conv14 = bias([128], 		name="B_conv10")
	B_fc1    = bias([1024], 	name="B_fc1")
	B_fc2    = bias([512],  	name="B_fc2")
	B_fc3    = bias([256],  	name="B_fc3")
	B_fc4 	 = bias([128],		name="B_fc4")
	B_out    = bias([n_outputs],name="B_out")

	# DEFINING PilotNetV2 ARCHITECTURE
	# Input Image(width = 32, height = 32, RGB) ->
	# Normalization ->
	# Convolution 1  (5x5) -> Relu ->
	# Convolution 2  (5x5) -> Relu ->
	# Convolution 3  (5x5) -> Relu -> Max_Pool -> Normalize -> Dropout ->
	# Convolution 4  (5x5) -> Relu ->
	# Convolution 5  (5x5) -> Relu ->
	# Convolution 6  (3x3) -> Relu -> Max_Pool -> Normalize -> Dropout ->
	# Convolution 7  (3x3) -> Relu ->
	# Convolution 8  (3x3) -> Relu -> Normalize ->
	# Convolution 9  (3x3) -> Relu ->
	# Convolution 10 (3x3) -> Relu -> Max_Pool -> Normalize -> Dropout ->
	# Convolution 11 (2x2) -> Relu ->
	# Convolution 12 (2x2) -> Relu -> Normalize ->
	# Convolution 13 (2x2) -> Relu ->
	# Convolution 14 (2x2) -> Relu ->
	# Fully Connected Layer 1 (1024) -> Relu -> Dropout ->
	# Fully Connected Layer 2 (512)  -> Relu -> Dropout ->
	# Fully Connected Layer 3 (256)  -> Relu -> Dropout ->
	# Fully Connected Layer 4 (128)  -> Relu -> Dropout ->
	# Output -> Softmax -> Classification
	x = tf.reshape(x, shape=[-1, HEIGHT, WIDTH, 3])
	print("Input Size: {}" .format(x.get_shape()))

	# to normalize in hidden layers, add the normalization layer:
	# 1. right after fc or conv layers
	# 2. right after ReLu
	normalized = normalize(x, is_training=is_training)

	conv1 = conv2d(normalized, W_conv1, B_conv1, strides=1)
	conv1 = relu(conv1)

	conv2 = conv2d(conv1, W_conv2, B_conv2, strides=1)
	conv2 = relu(conv2)

	conv3 = conv2d(conv2, W_conv3, B_conv3, strides=1)
	conv3 = relu(conv3)
	conv3 = max_pool2d(conv3, 2)
	conv3 = normalize(conv3, is_training)
	conv3 = dropout(conv3, 0.5, is_training)

	conv4 = conv2d(conv3, W_conv4, B_conv4, strides=1)
	conv4 = relu(conv4)

	conv5 = conv2d(conv4, W_conv5, B_conv5, strides=1)
	conv5 = relu(conv5)

	conv6 = conv2d(conv5, W_conv6, B_conv6, strides=1)
	conv6 = relu(conv6)
	conv6 = max_pool2d(conv6, 2)
	conv6 = normalize(conv6, is_training)
	conv6 = dropout(conv6, 0.5, is_training)

	conv7 = conv2d(conv6, W_conv7, B_conv7, strides=1)
	conv7 = relu(conv7)

	conv8 = conv2d(conv7, W_conv8, B_conv8, strides=1)
	conv8 = relu(conv8)
	conv8 = normalize(conv8, is_training)

	conv9 = conv2d(conv8, W_conv9, B_conv9, strides=1)
	conv9 = relu(conv9)

	conv10 = conv2d(conv9, W_conv10, B_conv10, strides=1)
	conv10 = relu(conv10)
	conv10 = max_pool2d(conv10, strides=2)
	conv10 = normalize(conv10, is_training)
	conv10 = dropout(conv10, 0.5, is_training)

	conv11 = conv2d(conv10, W_conv11, B_conv11, strides=1)
	conv11 = relu(conv11)

	conv12 = conv2d(conv11, W_conv12, B_conv12, strides=1)
	conv12 = relu(conv12)
	conv12 = normalize(conv12, is_training)

	conv13 = conv2d(conv12, W_conv13, B_conv13, strides=1)
	conv13 = relu(conv13)

	conv14 = conv2d(conv13, W_conv14, B_conv14, strides=1)
	conv14 = relu(conv14)

	# Note: Do not add Normalize layer right before FC layers

	# flatten to 1 dimension for fully connected layers
	flat = tf.reshape(conv14, shape=[-1, W_fc1.get_shape().as_list()[0]])

	fc1 = relu(tf.matmul(flat, W_fc1) + B_fc1)
	fc1 = dropout(fc1, 0.5, is_training)

	fc2 = relu(tf.matmul(fc1, W_fc2) + B_fc2)
	fc2 = dropout(fc2, 0.5, is_training)

	fc3 = relu(tf.matmul(fc2, W_fc3) + B_fc3)
	fc3 = dropout(fc3, 0.5, is_training)

	fc4 = relu(tf.matmul(fc3, W_fc4) + B_fc4)

	# Note: Do not add Dropout layer before Output layer

	output = tf.matmul(fc4, W_out) + B_out	# output activation is SoftMax, but is handled in training

	return output
