import numpy as np
from keras.datasets import mnist
# %matplotlib inline #if used on Jupyter Notebook


class conv3x3:
	def __init__(self, num_filters):
		self.num_filters = num_filters
		self.filters = np.random.randn(num_filters, 3, 3) / 9  ##初始权重采用随机方式

	def iterate_regions(self, image):
		##获取图片大小与卷积迭代区域
		h, w = image.shape

		for i in range(h - 2):
			for j in range(w - 2):
				im_region = image[i:(i + 3), j:(j + 3)]
				yield im_region, i, j

	def forward(self, input):
		##卷积操作
		self.last_input = input
		h, w = input.shape
		output = np.zeros((h - 2, w - 2, self.num_filters))

		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

		return output
	# 输出：output


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  ##导入mnist库
conv = conv3x3(8)
output = conv.forward(train_images[0])
print(output.shape)  ##测试卷积层正确情况 To test the conv is correct
#Output: (26,26,8)

