import numpy as np

def relu_derivative(z):
	return np.where(z > 0, 1.0, 0.0)

class Layer:
	def __init__(self, previous_layer, size, activation):
		self.previous_layer = previous_layer
		self.activation = activation
		self.size = size
		self.last_input_batch = []
		self.z_batch = []
		if previous_layer is None:
			self.input_size = 30
		else:
			self.input_size = previous_layer.size

		self.perceptrons = np.random.randn(size, self.input_size + 1) * 0.1
		self.m = np.zeros_like(self.perceptrons)
		self.v = np.zeros_like(self.perceptrons)
		self.t = 0
		self.beta1 = 0.9
		self.beta2 = 0.999
		self.epsilon = 1e-8
	
	def softmax(self, z_values):
		z_values = np.array(z_values, dtype=np.float64)
		exp_z = np.exp(z_values - np.max(z_values))
		return exp_z / np.sum(exp_z)

	def set_weights(self, weights_matrix):
		self.perceptrons = weights_matrix

	def go_forward(self, inputs):
		inputs_with_bias = np.append(inputs, 1)
		self.last_input_batch.append(inputs_with_bias)
		if self.activation == "Relu":
			outputs = []
			z = np.dot(self.perceptrons, inputs_with_bias)
			self.z_batch.append(z)
			output = np.maximum(0, z)
			outputs.append(output)
			return outputs
		if self.activation == "softmax":
			z = np.dot(self.perceptrons, inputs_with_bias)
			return self.softmax(z)

	def backpropagation(self, delta, lr):
		self.t += 1
		input_batch = np.array(self.last_input_batch)
		last_input = np.mean(input_batch, axis=0)
		last_input = np.array(last_input, dtype=np.float64)

		for i in range(len(self.perceptrons)):
			# For simple descent gradient
			# self.perceptrons[i] -= lr * delta[i] * last_input

			# For Adam formules
			g_t = delta[i] * last_input # get the gradient
			self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_t # Actualise m
			self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g_t ** 2) # Actualise v

			m_hat = self.m[i] / (1 - self.beta1 ** self.t) # Correction of biasis, tend to 0, more iteration == less correction
			v_hat = self.v[i] / (1 - self.beta2 ** self.t)
			self.perceptrons[i] -= lr * m_hat / (np.sqrt(v_hat) + self.epsilon)	# Actualise weights

		if self.previous_layer is not None:
			weights_matrix = self.perceptrons[:, :-1]
			z_means = np.mean(self.previous_layer.z_batch, axis=0)
			delta_prev = np.dot(weights_matrix.T, delta) * relu_derivative(z_means)
			self.last_input_batch.clear()
			self.z_batch.clear()
			self.previous_layer.backpropagation(delta_prev, lr)
		else:
			self.last_input_batch.clear()
			self.z_batch.clear()
