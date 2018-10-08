import numpy as np
import cv2

def unpickle(file):		# official given function to open CIFAR-10 file
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def bufferToImg(img_buffer):	# converts CIFAR-10 buffer data to displayable image format
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

def h_flip(src_img):	# horizontally flip the images
	return cv2.flip(src_img, 1)


def one_hot(class_id):		# convert class id's to one-hot format
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


def process_trainingData():	# save training data into np.array format and randomizing order
	training_data = []

	for batch_num in range(1,6):

		data = unpickle("./data/data_batch_{}" .format(batch_num))

		data_len = 10000
		for i in range(data_len):
			image = bufferToImg(data[b'data'][i])
			label = one_hot(data[b'labels'][i])
			training_data.append([image,label])

	np.random.shuffle(training_data)
	np.save("training_data.npy", training_data)


def process_testData():	# save testing data into np.array format and randomizing order
	data = unpickle("./data/test_batch")

	test_data = []

	data_len = 10000
	for i in range(data_len):
		image = bufferToImg(data[b'data'][i])
		label = one_hot(data[b'labels'][i])
		test_data.append([image,label])

	np.random.shuffle(test_data)
	np.save("test_data.npy", test_data)


def data_augment(data_set):	# augmenting data set
	data_aug = []
	for data_pt in data_set:
		data_aug.append([data_pt[0], data_pt[1]])
		data_aug.append([h_flip(data_pt[0]), data_pt[1]])

	np.random.shuffle(data_aug)
	print(len(data_aug))
	return data_aug


def main():
	training_data = np.load("training_data.npy")
	testing_data = np.load("test_data.npy")

	training_data_aug = data_augment(training_data)
	testing_data_aug = data_augment(testing_data)

	np.save("training_data.npy", testing_data_aug)
	np.save("testing_data.npy", testing_data_aug)


main()