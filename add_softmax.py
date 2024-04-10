import numpy as np
from keras.datasets import mnist



class conv3x3:
	def __init__(self, num_filters):
		self.num_filters = num_filters
		self.filters = np.random.randn(num_filters, 3, 3) / 9
		##The initial weights use the random way

	def iterate_regions(self, image):
		##Get the image size and convolution iteration area
		h, w = image.shape

		for i in range(h - 2):
			for j in range(w - 2):
				im_region = image[i:(i + 3), j:(j + 3)]
				yield im_region, i, j

	def forward(self, input):
		##cnn operation
		self.last_input = input
		h, w = input.shape
		output = np.zeros((h - 2, w - 2, self.num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

		return output
	# output


class MaxPool2:
	##池化操作，进一步压缩图片数据集提及
	##2x2最大值滤波器提取特征值
	def iterate_regions(self, image):

		h, w, _ = image.shape
		new_h = h // 2
		new_w = w // 2

		for i in range(new_h):
			for j in range(new_w):
				im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
				yield im_region, i, j

	def forward(self, input):

		self.last_input = input
		h, w, num_filters = input.shape
		output = np.zeros((h // 2, w // 2, num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.amax(im_region, axis=(0, 1))  ##numpy库指令直接调用选取每个2x2区域最大值

		return output


class Softmax:
	##Full neural network direct classification
	##Find the probability and classify it according to the probability (10 categories: 0-9)
	def __init__(self, input_len, nodes):
		##input_len (data size after Pooling: 13x13x8); nodes = 10, representing 10 classification nodes
		self.weights = np.random.randn(input_len, nodes) / input_len
		self.biases = np.zeros(nodes)

	def forward(self, input):
		input = input.flatten()  ##flatten the input and convert to one-dimensional data
		input_len, nodes = self.weights.shape
		totals = np.dot(input, self.weights) + self.biases
		exp = np.exp(totals)
		return exp / np.sum(exp, axis=0)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
##Import mnist library
conv = conv3x3(8)  ##cnn layer object
##Use 1000 test images
test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv = conv3x3(8)  ##28x28x1->26x26x8
pool = MaxPool2()  ##26x26x8->13x13x8
softmax = Softmax(13 * 13 * 8, 10)
##Data size and classification node selection


def forward(image, label):
	##image corresponds to the image entered into the recognition, label corresponds to 13x13x8
	out = conv.forward((image / 255) - 0.5)
	##Pixel value normalization [-0.5,0.5]
	out = pool.forward(out)
	out = softmax.forward(out)
	##Probability calculation results

	loss = -np.log(out[label])
	ac = 1 if np.argmax(out) == label else 0
	##Check the correctness of the classification result, 1 is true, 0 is false
	return out, loss, ac


print("Mnist CNN detect init!")
loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
	##Without pre-training, accuracy calculation is performed directly on the test data set.
	_, l, ac = forward(im, label)
	loss += l
	num_correct += ac  ##Accumulate the correct number of statistics
	##Statistics are performed every 100 steps
	if i % 100 == 99:
		print('[Steps %d] passed 100 photos：avg loss：%.3f|accuracy: %d%%' %
			(i + 1, loss / 100, num_correct))
		loss = 0
		num_correct = 0