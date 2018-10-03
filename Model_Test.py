import numpy as np
import tensorflow as tf
import cv2
import time
from Model_Arch import Model


tf.reset_default_graph()

WIDTH = 32
HEIGHT = 32
n_outputs = 10
pool_s = 2

x = tf.placeholder("float", [None, HEIGHT, WIDTH])
model = tf.nn.softmax(Model(x, WIDTH, HEIGHT, n_outputs, is_training=False))

loader = tf.train.Saver()

print("Loading Data")
test_data = np.load("test_data.npy")

# model information
version = "1"
model_name = "./Model_Data/CNN_v{}" .format(version)


def ID2Label(id):
	if (id == 0):
		return "airplane"
	elif (id == 1):
		return "automobile"
	elif (id == 2):
		return "bird"
	elif (id == 3):
		return "cat"
	elif (id == 4):
		return "deer"
	elif (id == 5):
		return "dog"
	elif (id == 6):
		return "frog"
	elif (id == 7):
		return "horse"
	elif (id == 8):
		return "ship"
	elif (id == 9):
		return "truck"




def main():
	last_time = time.time()

	training_variables = tf.trainable_variables()	# getting list of trainable variables defined in the model

	print("----- Loading Weights -----")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True	# allow dynamic allocation of memory
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		# load up saved model
		graph = tf.get_default_graph()
		W_test = graph.get_tensor_by_name("W_conv1:0")
		B_test = graph.get_tensor_by_name("B_conv1:0")
		loader.restore(sess, model_name)


		i = 0
		running_error = 0
		for screen in test_data:
			# making prediction
			# prediction = model.eval({x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]
			prediction = sess.run(model, feed_dict={x: screen[0].reshape(-1, HEIGHT, WIDTH)})[0]
			# print(ID2Label(np.argmax(prediction)))

			# adding prediction onto test photos
			display_img = cv2.resize(test_data[i][0], (WIDTH*10, HEIGHT*10))
			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(display_img,ID2Label(np.argmax(prediction)), (50, 50), font, 2, (0,0,255), 3)
			cv2.imshow("Model Test", display_img)

			i += 1
			if (i == len(test_data) - 1):
				cv2.destroyAllWindows()
				break
			if cv2.waitKey(25) & 0xFF == ord('q'):
					cv2.destroyAllWindows()
					break
			time.sleep(3)

		print("Average difference: {}" .format(running_error/i))


if __name__ == "__main__":
	main()



