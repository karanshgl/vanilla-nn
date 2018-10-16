from scipy import signal
import numpy as np
from activations import Activation

a = np.array([[[[1,2,3],[4,5,6],[7,8,9]]],[[[9,8,7],[6,5,4],[3,2,1]]]])
# k = np.array([[[1,1,1],[1,1,1],[1,1,1]]])
# print(a.shape, k.shape)
# print(signal.convolve(a, k, mode='valid')[0])

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
		kernel_size = (1, *kernel_size)
		self.weights = [np.random.normal(0.0, 1.0/np.sqrt(num_weights), kernel_size) for filter in range(filters)]
		self.bias = [np.random.normal(0.0, 1.0/np.sqrt(num_weights), (1, 1)) for filter in range(filters)]

		self.padding = padding
		self.stride = stride
		self.output_units = []

		self.activation = Activation(activation)

	def forward_pass(self, input_units):

		outputs_per_batch = []
		for batch in input_units:
			output_units = [signal.convolve(batch, filter, mode = self.padding) for filter in self.weights]
			output_units = np.concatenate(output_units, axis=0)
			outputs_per_batch.append(output_units[np.newaxis])

		self.output_units = np.concatenate(outputs_per_batch, axis=0)
		activated_output_units = self.activation.function(np.array(self.output_units))
		return activated_output_units

	def backward_pass(self, input_units, grad_activated_output):
		pass


layer = Conv2D(5,(3,3))
print(a.shape)
print(layer.forward_pass(a).shape)
