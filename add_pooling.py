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
		##Conv operations
		self.last_input = input
		h, w = input.shape
		output = np.zeros((h - 2, w - 2, self.num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

		return output
	# output


class MaxPool2:
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

	def forward(self, input):

		self.last_input = input
		h, w, num_filters = input.shape
		output = np.zeros((h // 2, w // 2, num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.amax(im_region, axis=(0, 1))
		##The numpy library instruction directly calls to select the maximum value of each 2x2 area

		return output

##test code


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  ##导入mnist库
conv = conv3x3(8)  ##Conv layer's object
Pool = MaxPool2()  ##Pooling layer's object
output = conv.forward(train_images[0]) ##The output of cnn layer is used as the input of the pooling layer
output = Pool.forward(output)
print(output.shape)
## Output:(13,13,8)