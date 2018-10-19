import numpy as np
from im2col import *

class Maxpool:

	def __init__(self, kernel_size, stride = 1, padding ='same'):

		self.kernel_size = kernel_size
		self.stride = stride
		self.output_units = None
		self.padding = (kernel_size[0]-1)//2 if padding == 'same' else 0

	def forward_pass(self, input_units):
		"""
		Params:
		input_units = the input nodes

		Returns:
		The output_units
		"""
		N, C, H, W = input_units.shape
		pool_height, pool_width = self.kernel_size[0], self.kernel_size[1]
		stride, pad = self.stride, self.padding

		# assert (H - pool_height) % stride == 0, 'Invalid height'
		# assert (W - pool_width) % stride == 0, 'Invalid width'

		out_height = (H - pool_height + 2*pad) // stride + 1
		out_width = (W - pool_width + 2*pad) // stride + 1

		x_split = input_units.reshape(N * C, 1, H, W)
		self.x_cols = im2col_indices(x_split, pool_height, pool_width, pad, stride)
		self.x_cols_argmax = np.argmax(self.x_cols, axis=0)

		x_cols_max = self.x_cols[self.x_cols_argmax, np.arange(self.x_cols.shape[1])]
		out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

		# cache = (x, x_cols, x_cols_argmax, pool_param)
		return out



	def backward_pass(self, input_units, grad_activated_output):
		"""
		"""
		x, x_cols, x_cols_argmax = input_units, self.x_cols, self.x_cols_argmax
		N, C, H, W = x.shape
		pool_height, pool_width = self.kernel_size[0], self.kernel_size[1]
		stride, pad = self.stride, self.padding

		dout_reshaped = grad_activated_output.transpose(2, 3, 0, 1).flatten()
		dx_cols = np.zeros_like(x_cols)
		dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
		dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width, pad, stride)
		dx = dx.reshape(x.shape)

		return dx

	def run(self, input_units):
		return self.forward_pass(input_units)


	def update(self, learning_rate):
		pass