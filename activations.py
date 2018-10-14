import numpy as np
from scipy.special import expit as sigmoid

class Activation:

	def __init__(self, activation):

		self.function = lambda x: x
		self.derivative = lambda x : 1

		if activation is not None:

			if activation == "sigmoid":
				self.function = lambda x : sigmoid(x)
				self.derivative = lambda x : sigmoid(x)*(1-sigmoid(x))
