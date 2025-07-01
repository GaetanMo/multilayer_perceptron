from .perceptron import Perceptron
import numpy as np

def relu_derivative(z):
	return np.where(z > 0, 1.0, 0.0)

class Layer:
	def __init__(self, previous_layer, size, activation):
		self.previous_layer = previous_layer
		self.activation = activation
		self.size = size
		self.last_input_batch = []

		if previous_layer is None:
			self.input_size = 30
		else:
			self.input_size = previous_layer.size

		self.perceptrons = []
		for _ in range (size):
			p = Perceptron()
			self.perceptrons.append(p)

	def softmax(self, z_values):
		z_values = np.array(z_values)
		exp_z = np.exp(z_values - np.max(z_values))
		return exp_z / np.sum(exp_z)

	def set_weights(self, weights_matrix):
		for i, perceptron in enumerate(self.perceptrons):
			perceptron.set_weight(weights_matrix[i])

	def go_forward(self, inputs):
		inputs_with_bias = np.append(inputs, 1)
		self.last_input_batch.append(inputs_with_bias)
		if self.activation == "sigmoid":
			outputs = []
			for i in range (self.size):
				output = self.perceptrons[i].activate_relu(inputs_with_bias)
				outputs.append(output)
			return outputs

		if self.activation == "softmax":
			z_values = [p.compute_z(inputs_with_bias) for p in self.perceptrons]
			return self.softmax(z_values)

	def backpropagation(self, delta, lr):
		input_batch = np.array(self.last_input_batch)
		last_input = np.mean(input_batch, axis=0)
		for i, perceptron in enumerate(self.perceptrons):
			weight_update = -lr * delta[i] * last_input
			perceptron.set_weight(perceptron.weight + weight_update)

		if self.previous_layer is not None:
			weights_matrix = np.array([p.weight[:-1] for p in self.perceptrons])
			z_means = np.array([np.mean(p.z_batch) for p in self.previous_layer.perceptrons])
			delta_prev = np.dot(weights_matrix.T, delta) * relu_derivative(z_means)
			self.last_input_batch.clear()
			for p in self.perceptrons:
				p.z_batch.clear()
			self.previous_layer.backpropagation(delta_prev, lr)
		else:
			self.last_input_batch.clear()
			for p in self.perceptrons:
				p.z_batch.clear()