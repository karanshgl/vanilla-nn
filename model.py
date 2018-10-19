import numpy as np
from dense_layer import Dense

class Model:

	def __init__(self, learning_rate = 0.001, batch_size = 1, epochs = 100, optimizer = None):
		"""
		Params:
		learning_rate = Learning Rate of the Model
		"""
		self.layers = []
		self.units = [None]
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.optimizer = None


	def generate_batches(self, X, Y):

		# print(X.shape, Y.shape)
		indices = np.arange(X.shape[0])
		np.random.shuffle(indices)
		# concat = np.concatenate((X,Y), axis = 1)
		# X,Y = concat[:,:-1], concat[:,-1]
		X, Y = X[indices], Y[indices]
		batch_count = X.shape[0]//self.batch_size + 1
		for i in range(batch_count):
			start = i*self.batch_size
			end = (i+1)*self.batch_size if ((i+1)< batch_count) else X.shape[0]
			yield X[start:end], Y[start:end]



	def train(self, X, Y):
		
		for epoch in range(self.epochs):

			loss = 0
			for x,y in self.generate_batches(X,Y):
				self.units[0] = x
				for i, layer in enumerate(self.layers):
					self.units[i+1] = layer.forward_pass(self.units[i])

				error = self.units[-1]-y.reshape((len(y), 1))
				loss += np.square(error).sum()/2

				learning_rate = self.learning_rate if self.optimizer is None else self.optimizer.step_size(self.learning_rate, error.sum())

				for layer, unit in zip(reversed(self.layers), reversed(self.units[:-1])):
					error = layer.backward_pass(unit,error)
					layer.update(learning_rate)
				
			print("Loss: ", loss/X.shape[0])


	def test(self, X, Y):

		loss = 0
		for x,y in self.generate_batches(X,Y):
			unit = x
			for i,layer in enumerate(self.layers):
				unit = layer.run(unit)

			loss += np.square(unit-y).sum()/2
		
		print("Loss: ", loss/X.shape[0])

	def add(self, layer):
		self.layers.append(layer)
		self.units.append(None)

