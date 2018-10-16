import numpy as np
from activations import Activation
from maxpool import Maxpool

# a = np.array([[[[1,2,3],[4,5,6],[7,8,9]]],[[[9,8,7],[6,5,4],[3,2,1]]]])
a = np.ones((64,3,32,32))
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
		self.kernel_size = kernel_size
		self.weights = np.array([np.random.normal(0.0, 1.0/np.sqrt(num_weights), kernel_size) for filter in range(filters)])
		self.bias = np.array([np.random.normal(0.0, 1.0/np.sqrt(num_weights), (1, 1)) for filter in range(filters)])

		self.padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2) if padding == 'same' else (0,0)
		self.stride = stride
		self.output_units = []

		self.activation = Activation(activation)


	def conv2d(self, image, kernel):

		new_row = ((image.shape[0]-self.kernel_size[0] + 2*self.padding[0])//self.stride) + 1
		new_col = ((image.shape[1]-self.kernel_size[1] + 2*self.padding[1])//self.stride) + 1

		output = np.zeros((new_row, new_col))
		image = np.pad(image, self.padding, 'constant')

		old_row, old_col = image.shape
		nr, nc = 0,0
		for r in range(0, old_row - self.kernel_size[0] + 1, self.stride):
			nc = 0
			for c in range(0, old_col - self.kernel_size[1] + 1, self.stride):
				window = image[r : r + self.kernel_size[0], c : c + self.kernel_size[1]]
				output[nr,nc] = (np.multiply(window,kernel)).sum()
				nc += 1
			nr += 1

		return output



	def forward_pass(self, input_units):

		output_units = []
		for batch in input_units:
			batch = batch.sum(axis=0)
			depth_outputs = [self.conv2d(batch, kernel) for kernel, bias in zip(self.weights, self.bias)]
			output_batch = np.array(depth_outputs)
			output_units.append(output_batch)

		self.output_units = np.array(output_units)
		activated_output_units = self.activation.function(self.output_units)

		return activated_output_units

	def backward_pass(self, input_units, grad_activated_output):

		self.grad_weights = [np.zeros(kernel.shape) for kernel in self.weights]
		self.grad_bias = [np.zeros((1,1)) for bias in self.bias]
		# grad_activated_input = np.zeros(input_units.shape)

		total_grad_input = []
		for batch in range(input_units.shape[0]):
			grad_input = []
			image = np.pad(input_units[batch], ((0,0), self.padding, self.padding), mode = 'constant')
			grad_activated_input = np.zeros(image.shape)
			for i in range(len(self.weights)):
				# For Each Kernel and Bias

				old_row, old_col = image.shape[1], image.shape[2]
				eff_output_index = i
				nr, nc = 0,0
				for r in range(0, old_row - self.kernel_size[0] + 1, self.stride):
					nc = 0
					for c in range(0, old_col - self.kernel_size[1] + 1, self.stride):
						grad_output_units = np.multiply(grad_activated_output[batch,eff_output_index,nr,nc],\
							self.activation.derivative(self.output_units[batch, eff_output_index, nr, nc]))

						grad_activated_input[:,r : r + self.kernel_size[0], c : c + self.kernel_size[1]] +=\
						 	np.multiply(grad_output_units,self.weights[i])

						self.grad_weights[i] += \
						 np.multiply(image[:,r : r + self.kernel_size[0], c : c + self.kernel_size[1]], grad_output_units).sum(axis=0)
						self.grad_bias[i] += grad_output_units.sum()

						nc += 1
					nr += 1
				
				# Add the image
				# grad_input.append(grad_activated_input)
			grad_input = np.array(grad_activated_input[:,self.padding[0]:-self.padding[0],self.padding[1]:-self.padding[1]])
			total_grad_input.append(grad_input)

		self.grad_bias = np.array(self.grad_bias)
		self.grad_weights = np.array(self.grad_weights)

		return np.array(total_grad_input)

	def update(self, learning_rate):
		self.weights -= learning_rate*self.grad_weights
		self.bias 	 -= learning_rate*self.grad_bias


layer = Conv2D(1,(5,5), stride = 4)
print(a.shape)
o = layer.forward_pass(a.copy())
print(o.shape)
print(layer.backward_pass(a,o).shape)
layer.update(0.01)
maxpool = Maxpool((3,3), stride = 2)
o2 = maxpool.forward_pass(o)
print(o2.shape)
maxpool.backward_pass(o,o2)

