import numpy as np
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def bufferToImg(img_buffer):
	img = []
	sq_img_dim = int(len(img_buffer)/3)

	# Parsing out R, G, B channels from image data buffer
	R = img_buffer[:sq_img_dim]
	G = img_buffer[sq_img_dim:sq_img_dim*2]
	B = img_buffer[-sq_img_dim:]

	# Reformatting buffer data into proper image format
	index = 0
	img_dim = int(np.sqrt(len(img_buffer)/3))
	for i in range(img_dim):
		img_row = []
		for j in range(img_dim):
			img_row.append([R[index], G[index], B[index]])
			index += 1
		img.append(img_row)

	return np.array(img)


def one_hot(class_id):
	one_hot_class = [0,0,0,0,0,0,0,0,0,0]

	if (class_id == 0):
		one_hot_class = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	elif (class_id == 1):
		one_hot_class = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
	elif (class_id == 2):
		one_hot_class = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
	elif (class_id == 3):
		one_hot_class = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
	elif (class_id == 4):
		one_hot_class = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
	elif (class_id == 5):
		one_hot_class = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	elif (class_id == 6):
		one_hot_class = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
	elif (class_id == 7):
		one_hot_class = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
	elif (class_id == 8):
		one_hot_class = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
	elif (class_id == 9):
		one_hot_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

	return one_hot_class


def process_trainingData():
	training_data = []

	for batch_num in range(1,6):

		data = unpickle("./data/data_batch_{}" .format(batch_num))
		print(data.keys())

		data_len = 10000
		for i in range(data_len):
			image = bufferToImg(data[b'data'][i])
			label = one_hot(data[b'labels'][i])
			training_data.append([image,label])

	print(len(training_data))
	np.save("training_data.npy", training_data)


def process_testData():
	data = unpickle("./data/test_batch")

	test_data = []

	data_len = 10000
	for i in range(data_len):
		image = bufferToImg(data[b'data'][i])
		label = one_hot(data[b'labels'][i])
		test_data.append([image,label])

	np.save("test_data.npy", test_data)


def main():
	process_trainingData()


main()