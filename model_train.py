import tensorflow as tf
from Model_Arch import Model
import matplotlib.pyplot as plt
import numpy as np
import time


### HELPER FUNCTIONS ###
def k_fold_splitter(training_data, k=5):	# splitting training data into k equal parts
	print("K-Fold Split with K={}" .format(k))
	fold_len = int(len(training_data)/k)

	split_data = []
	for i in range(k):
		start_index = i*fold_len
		split_data.append(training_data[start_index:(start_index + fold_len - 1)])

	return split_data

def k_fold_selector(split_data, index):	# sectioning out train and test set
	print("K-Fold Selector with Index={}" .format(index))
	test_data = split_data[index]

	train_data = []
	for i in range(len(split_data)):
		if (i != index):
			for item in split_data[i]:
				train_data.append(item)

	train_x = [i[0] for i in train_data]
	train_y = [i[1] for i in train_data]
	test_x 	= [i[0] for i in test_data]
	test_y 	= [i[1] for i in test_data]

	return train_x, train_y, test_x, test_y

def early_stop(data, strikes, strikeout, threshold, monitor="validation"):	# checks the condition on when to finish training
	delta_data = data[1] - data[0]

	if (monitor == "validation"):
		# if the change in loss is less than [tolerance]% or if the loss increased, add a strike
		if (abs(delta_data) < abs(threshold*data[0]) or delta_data > 0):
			strikes += 1
			if (strikes == strikeout):
				strikes = -1
		else:
			strikes = 0

	elif (monitor == "accuracy"):
		# if the accuracy decreases
		if (delta_data < 0):
			strikes += 1
			if (strikes == strikeout):
				strikes = -1
		else:
			strikes = 0

	return strikes

def cyclical_lr(epoch, amplitude, period):	#todo make this work with decay LR
	return np.sin(epoch*2*np.pi/period)*amplitude


### GLOBALS ###
# Loading Data
print("Loading Data...")
loaded_data = np.load("training_data.npy")

# K-Fold Separation of Training & Test Set
print("Separating Data into Training and Test Sets...")
k_splits = 5
k_index  = 2
k_split_data = k_fold_splitter(loaded_data, k_splits)
train_x, train_y, test_x, test_y = k_fold_selector(k_split_data, index=k_index)
print("[X] Train Size: {}. Test Size: {}".format(len(train_x), len(test_x)))
print("[Y] Train Size: {}. Test Size: {}".format(len(train_y), len(test_y)))

												# Value Increase Effect ************************************************
batch_size = 64  								# -converge into sharper minima, less iterations
n_epochs = 300									# -increase training time, lower training loss, higher risk overfitting
initial_learning_rate = 4e-5					# -faster training time, bigger gradient jumps
epsilon = 2e-5									# -smaller weight updates
decay_rate = 0.85								# -less weight decay (smaller gradient jumps)
epochs_per_decay = 50							# -more frequent learning rate decay (smaller and smaller gradient jumps)
												# **********************************************************************
steps_per_epoch = len(train_x)/batch_size
n_outputs = 10
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, epochs_per_decay*steps_per_epoch,
										   decay_rate, staircase=True, name="LR_Decaying")

# Input Image Dimensions
WIDTH  = 32
HEIGHT = 32
# Input Tensor Placeholders
x = tf.placeholder("float", [None, HEIGHT, WIDTH, 3])
y = tf.placeholder("float", [None, n_outputs])

# Model Information
version = "1"
model_name = "./Model_Data/CNN_v{}" .format(version)


### TRAINING FUNCTION ###
# Specifications
# Model: 				PilotNetV2
# Cost Function: 		Mean Squared Loss with L2 Loss Weight Penalization
# Optimizer: 			AdamOptimizer
# Learning Rate Mod: 	Exponential Decay
# Special Functions: 	Early Stopping, K-Fold Cross-Validation
def ConvNN_Train(x):
	# Training Operations
	prediction = Model(x, WIDTH, HEIGHT, n_outputs, is_training=True)
	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))\
		+ tf.add_n([tf.nn.l2_loss(variable) for variable in training_variables if "B" not in variable.name])*learning_rate
	correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct, "float"))

	# Optimizer with Normalization
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon).minimize(cost, global_step=global_step)

	# Plot to Monitor Loss
	x_max = n_epochs
	x_scale = n_epochs/20
	y_max = 1
	y_scale = .1
	plt.figure(figsize=(15,8))
	plt.axis([0, x_max, 0, y_max])
	plt.grid(True)
	plt.xticks(np.arange(0, x_max, x_scale))
	plt.yticks(np.arange(0, y_max, y_scale))
	plt.xlabel("Epoch Number")
	plt.ylabel("Accuracy")
	plt.title("Training and Testing Accuracy ")

	# Learning Rate Monitor
	graph = tf.get_default_graph()
	lr_test = graph.get_tensor_by_name("LR_Decaying:0")


	start = time.time()	# keeping track of total training time


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# enabling dynamic allocation of GPU memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		# Early Stop Variables
		strikes = 0		# metric for the early stopping. each time the delta loss falls under desired threshold, add a strike
		E_data_prev = 0
		is_saved = False

		# Training Model
		# Prefix Guide: "E" refers to epoch scope, while "B" refers to batch scope
		print("Training Starting!")
		for epoch in range(n_epochs):
			# Feeding in Training Data
			E_train_loss = 0
			E_train_acc = 0
			for batch in range(int(len(train_x)/batch_size)):	# feeding in training batches
				B_train_x = train_x[batch*batch_size:min((batch+1)*batch_size, len(train_x)-1)]
				B_train_y = train_y[batch*batch_size:min((batch+1)*batch_size, len(train_y)-1)]

				opt, B_train_loss, B_train_acc = sess.run([optimizer, cost, accuracy], feed_dict={x: B_train_x, y: B_train_y})
				E_train_loss += B_train_loss
				E_train_acc += B_train_acc
			E_train_acc = E_train_acc/int(len(train_x)/batch_size)

			# Feeding in Validation Data
			E_val_loss = 0
			E_val_acc = 0
			for batch in range(int(len(test_x)/batch_size)):	# feeding in testing batches
				B_test_x = test_x[batch*batch_size:min((batch+1)*batch_size, len(test_x)-1)]
				B_test_y = test_y[batch*batch_size:min((batch+1)*batch_size, len(test_y)-1)]

				B_test_loss, B_val_acc = sess.run([cost, accuracy], feed_dict={x: B_test_x, y: B_test_y})
				E_val_loss += B_test_loss
				E_val_acc += B_val_acc
			E_val_acc = E_val_acc/int(len(test_x)/batch_size)

			# Statistics Per Epoch
			print("\nEpoch {}/{}." .format(epoch+1, n_epochs))
			print("Training Loss  : {}" .format(E_train_loss))
			print("Validation Loss: {}" .format(E_val_loss))
			print("Training Acc   : {}" .format(E_train_acc))
			print("Validation Acc : {}" .format(E_val_acc))
			print("Learning Rate: {}" .format(lr_test.eval()))


			pt_color = "blue"	# blue colored pt for decreasing loss

			# Early Stopping Check
			if (epoch >= 1):
				strikes = early_stop([E_data_prev, E_val_acc], strikes, 5, 0.005, "accuracy")
				E_data_prev = E_val_acc
				print("Strikes: {}" .format(strikes))
				if (strikes == 1):	# saving model right when validation loss starts to increase
					print("Saving Model Checkpoint")
					saver.save(sess, model_name)
					pt_color = "red"	# red colored pt for increasing loss
				elif (strikes > 1):
					pt_color = "red"
				else:
					pt_color = "blue"
				if (strikes == -1):	# strikeout condition
					print("Early Stop at Epoch:{}/{}" .format(epoch, n_epochs))
					is_saved = True
					end = time.time()
					print("\nTraining Done! Time Elapsed: {} minutes".format((end - start) / 60.0))
					plt.show()
					break

			plt.scatter(epoch, E_val_acc, c=pt_color)
			plt.pause(0.01)
			plt.scatter(epoch, E_train_acc, c="green")
			plt.pause(0.01)

		end = time.time()
		print("\nTraining Done! Time Elapsed: {} minutes" .format((end - start)/60.0))
		plt.show()

		# Saving Model
		if (not is_saved):
			print("Saving Model: {}" .format(model_name))
			saver.save(sess, model_name)
		else:
			print("Model Saved: {}" .format(model_name))

		plt.close("all")



if __name__ == '__main__':
	ConvNN_Train(x)

