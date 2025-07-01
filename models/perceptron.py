import numpy as np

class Perceptron:
	def __init__(self):
		self.weight = None
		self.z = 0
		self.z_batch = []
	def set_weight(self, weight):
		self.weight = np.array(weight)

	def compute_z(self, inputs):
		inputs = np.array(inputs)
		self.z = np.dot(inputs, self.weight)
		self.z_batch.append(self.z)
		return self.z

	def activate_relu(self, inputs):
		inputs = np.array(inputs)
		self.z = np.dot(inputs, self.weight)  # produit pondéré
		self.z_batch.append(self.z)
		output = np.maximum(0, self.z)        # activation ReLU
		return output