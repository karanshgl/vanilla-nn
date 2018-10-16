import numpy as np


class Maxpool:

	def __init__(self, kernel_size, stride = 1, padding ='same'):

		self.kernel_size = kernel_size
		self.stride = stride
		self.output_units = None
		self.padding = ((kernel_size[0]-1)//2, (kernel_size[1]-1)//2) if padding == 'same' else (0,0)

	def forward_pass(self, input_units):
		"""
		Params:
		input_units = the input nodes

		Returns:
		The output_units
		"""
		new_row = ((input_units.shape[2]-self.kernel_size[0] + 2*self.padding[0])//self.stride) + 1
		new_col = ((input_units.shape[3]-self.kernel_size[1] + 2*self.padding[1])//self.stride) + 1

		self.masks = np.zeros((input_units.shape[0], input_units.shape[1], input_units.shape[2] + self.padding[0], input_units.shape[3] + self.padding[1]))
		self.output_units = np.zeros((input_units.shape[0], input_units.shape[1], new_row, new_col))


		for b,batch in enumerate(input_units):
			batch = np.pad(batch, ((0,0), self.padding, self.padding), mode = 'constant')
			for d,depth in enumerate(batch):
				nr = 0
				nc = 0
				for r in range(0, depth.shape[0] - self.kernel_size[0] + 1, self.stride):
					nc = 0
					for c in range(0, depth.shape[1] - self.kernel_size[1] + 1, self.stride):
						self.masks[b, d, r:r + self.kernel_size[0], c:c + self.kernel_size[1]] = \
						depth[r:r + self.kernel_size[0], c:c + self.kernel_size[1]].max(keepdims = True) ==\
						depth[r:r + self.kernel_size[0], c:c + self.kernel_size[1]]
						self.output_units[b,d,nr,nc] = depth[r:r + self.kernel_size[0], c:c + self.kernel_size[1]].max()
						nc += 1
					nr += 1

		# self.output_units = input_units*self.masks
		return self.output_units


	def backward_pass(self, input_units, grad_activated_output):
		"""
		"""
		input_units = np.pad(input_units, ((0,0), (0,0), self.padding, self.padding), mode = 'constant')
		grad_activated_mask = np.zeros(input_units.shape)
		for b,batch in enumerate(input_units):
			for d,depth in enumerate(batch):
				nr = 0
				nc = 0
				for r in range(0, depth.shape[0] - self.kernel_size[0] + 1, self.stride):
					nc = 0
					for c in range(0, depth.shape[1] - self.kernel_size[1] + 1, self.stride):
						grad_activated_mask[b, d, r:r + self.kernel_size[0], c:c + self.kernel_size[1]] += \
						self.masks[b, d, r:r + self.kernel_size[0], c:c + self.kernel_size[1]]*self.output_units[b,d,nr,nc] 
						nc += 1
					nr += 1

		return grad_activated_mask


	def update(self, learning_rate):
		pass