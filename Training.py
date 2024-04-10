import numpy as np
from keras.datasets import mnist


class conv3x3:
	##The first layer of neural network: convolution extraction of feature values
	def __init__(self, num_filters):
		##num_filters represents the number of convolution kernels
		self.num_filters = num_filters
		self.filters = np.random.randn(num_filters, 3, 3) / 9
	##Initially, Gaussian distribution weights are used in a random manner
	# (/9 In order to improve efficiency, the occurrence of maximum values is screened out)

	def iterate_regions(self, image):
		##Get the image size and convolution iteration area
		h, w = image.shape
		# Read image size
		for i in range(h - 2):
			for j in range(w - 2):
				im_region = image[i:(i + 3), j:(j + 3)]
				yield im_region, i, j

	##Get the data content of the image participating in convolution
	def forward(self, input):
		##cnn operation
		self.last_input = input
		h, w = input.shape
		output = np.zeros((h - 2, w - 2, self.num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))  ##3x3 * 8x3x3

		return output

	# output（three dimensional matrix）

	def backprop(self, d_L_d_out, learn_rate):
		##Optimize weights based on gradient calculations
		##d_L_d_out is the post-gradient test
		d_L_d_filters = np.zeros(self.filters.shape)
		##Update correction weight to 0

		for im_region, i, j in self.iterate_regions(self.last_input):
			for f in range(self.num_filters):
				d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
			##Gradient descent method application

		# Update convolution weight filters
		self.filters -= learn_rate * d_L_d_filters
		return None


class MaxPool2:
	##The second layer of the neural network, pooling operation, further reduces the amount of data and extracts feature values.
	##Pooling operation to further compress the image data set mentioned
	##2x2 maximum filter extracts feature values
	def iterate_regions(self, image):

		h, w, _ = image.shape
		new_h = h // 2
		new_w = w // 2

		for i in range(new_h):
			for j in range(new_w):
				im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
				yield im_region, i, j

	def backprop(self, d_L_d_out):
		##d_L_d_out is equivalent to finding the partial derivative for the output layer

		d_L_d_input = np.zeros(self.last_input.shape)  ##clear
		for im_region, i, j in self.iterate_regions(self.last_input):
			h, w, f = im_region.shape
			amax = np.amax(im_region, axis=(0, 1))

			for i2 in range(h):
				for j2 in range(w):
					for f2 in range(f):
						##If it is the maximum gradient, keep it
						if im_region[i2, j2, f2] == amax[f2]:
							d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

		return d_L_d_input

	def forward(self, input):
		# Find the maximum value in each area
		self.last_input = input
		h, w, num_filters = input.shape
		output = np.zeros((h // 2, w // 2, num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.amax(im_region, axis=(0, 1))
		##The numpy library instruction directly calls to select the maximum value of each 2x2 area

		return output


class Softmax:
	##The third layer of the neural network uses the full neural network to directly classify the probability distribution.
	##Find the probability and classify it according to the probability (10 categories: 0-9)
	def __init__(self, input_len, nodes):
		##input_len (data size after Pooling: 13x13x8); nodes = 10, representing 10 classification nodes
		self.weights = np.random.randn(input_len, nodes) / input_len
		self.biases = np.zeros(nodes)

	def forward(self, input):

		self.last_input_shape = input.shape
		input = input.flatten()
		##Convert to one-dimensional data
		self.last_input = input
		input_len, nodes = self.weights.shape

		totals = np.dot(input, self.weights) + self.biases
		self.last_totals = totals

		totals = np.dot(input, self.weights) + self.biases
		exp = np.exp(totals)
		return exp / np.sum(exp, axis=0)

	##Probability calculation

	def backprop(self, d_L_d_out, learn_rate):
		# optimize
		for i, grad in enumerate(d_L_d_out):
			if grad == 0:
				continue

			# e^totals cal
			t_exp = np.exp(self.last_totals)
			# e^totals' sum
			Sum = np.sum(t_exp)
			# totals gradient calculation about out[i]
			d_out_d_t = -t_exp[i] * t_exp / (Sum ** 2)
			d_out_d_t[i] = t_exp[i] * (Sum - t_exp[i]) / (Sum ** 2)
			# Gradient calculation of totals on weights/biases/input layer
			d_t_d_w = self.last_input
			d_t_d_b = 1
			d_t_d_inputs = self.weights

			# Loss gradient calculation for totals
			d_L_d_t = grad * d_out_d_t

			# Loss for gradient calculation of weights/biases/input layer
			d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
			d_L_d_b = d_L_d_t * d_t_d_b
			d_L_d_inputs = d_t_d_inputs @ d_L_d_t

			##Update weights
			self.weights -= learn_rate * d_L_d_w
			self.biases -= learn_rate * d_L_d_b
			return d_L_d_inputs.reshape(self.last_input_shape)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  ##导入mnist库
conv = conv3x3(8)  ##cnn object
##Use the required number of training images
train_images = train_images[:5000]
train_labels = train_labels[:5000]
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


def train(im, label, study=.005):
	out, loss, ac = forward(im, label)
	# Calculate gradient and optimize
	grad = np.zeros(10)  # Class 10 initialization
	grad[label] = -1 / out[label]

	grad = softmax.backprop(grad, study)
	grad = pool.backprop(grad)
	grad = conv.backprop(grad, study)

	return loss, ac


print("Mnist CNN detect Init!")

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(train_images, train_labels)):
	##Statistics are performed every 100 steps
	if i % 100 == 99:
		print(
			'[Steps %d] passed 100 photos：avg loss：%.3f|accuracy: %d%%' %
			(i + 1, loss / 100, num_correct)
		)
		loss = 0
		num_correct = 0
	l, ac = train(im, label)
	loss += l
	num_correct += ac
##Test model effect
print('\nStrart to test')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
	# Test the test set to simulate the model optimization effect
	_, l, ac = forward(im, label)
	loss += l
	num_correct += ac
	##Accumulate the correct number of statistics
num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Acc：', num_correct / num_tests)