import numpy as np
from activations import Activation
from maxpool import Maxpool
from im2col import *


class Conv2D:

	def __init__(self, filters, kernel_size, stride = 1, padding ='same', activation = None):
		"""
		Params:
		filters: Number of Filters
		kernel_size: shape of the kernel
		stride: the stride
		padding: valid or same
		activation: activation function
		"""
		self.filters = filters

		num_weights = kernel_size[0]*kernel_size[1]
		self.kernel_size = kernel_size
		self.weights = None
		self.bias = None

		self.padding = (kernel_size[0]-1)//2 if padding == 'same' else 0
		self.stride = stride
		self.output_units = []

		self.activation = Activation(activation)



	def forward_pass(self, input_units):

		if self.weights is None:
			self.weights = np.random.normal(0, 1.0/np.sqrt(self.filters), (self.filters, input_units.shape[1], self.kernel_size[0], self.kernel_size[1]))
		if self.bias is None:
			self.bias = np.random.normal(0, 1.0/np.sqrt(self.filters), (self.filters,1))

		N, C, H, W = input_units.shape
		num_filters,_, filter_height, filter_width = self.weights.shape
		stride, pad = self.stride, self.padding

		# Create output
		out_height = (H + 2 * pad - filter_height) // stride + 1
		out_width = (W + 2 * pad - filter_width) // stride + 1
		out = np.zeros((N, num_filters, out_height, out_width))

		self.x_cols = im2col_indices(input_units, filter_height, filter_width, pad, stride)
		# self.x_cols = im2col_cython(input_units, self.weights.shape[2], self.weights.shape[3], pad, stride)
		res = self.weights.reshape((self.weights.shape[0], -1)).dot(self.x_cols) + self.bias.reshape(-1, 1)

		out = res.reshape(self.weights.shape[0], out.shape[2], out.shape[3], input_units.shape[0])
		out = out.transpose(3, 0, 1, 2)

		self.output_units = out
		out = self.activation.function(out)

		return out

	def backward_pass(self, input_units, grad_activated_output):

		x, w, b, x_cols = input_units, self.weights, self.bias, self.x_cols
		stride, pad = self.stride, self.padding

		grad_output = grad_activated_output*self.activation.derivative(self.output_units)
		db = np.sum(grad_output, axis=(0, 2, 3))[:,np.newaxis]

		num_filters, _, filter_height, filter_width = w.shape
		dout_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(num_filters, -1)
		dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

		dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
		dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)

		self.grad_weights = dw 
		self.grad_bias = db
		return dx

	def update(self, learning_rate):
		self.weights -= learning_rate*self.grad_weights
		self.bias 	 -= learning_rate*self.grad_bias

	def run(self, input_units):
		return self.forward_pass(input_units)


# layer = Conv2D(1,(3,3), stride = 1)
# print(a.shape)
# o = layer.forward_pass(a.copy())
# print(o.shape)
# print(layer.backward_pass(a,o).shape)
# layer.update(0.01)
# maxpool = Maxpool((3,3), stride = 2)
# o2 = maxpool.forward_pass(o)
# print(o2.shape)

