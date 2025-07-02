from .layer import Layer
from .graph.drawing import draw_loss
from .graph.drawing import draw_accuracy
import numpy as np

graph_epoch = []
graph_losses_train = []
graph_losses_valid = []
graph_train_accuracy = []
graph_validation_accuracy = []
best_loss_validation = float('inf')
patience = 5
wait = 0

def early_stopping(loss_validation):
	global best_loss_validation, wait
	if loss_validation < best_loss_validation:
		best_loss_validation = loss_validation
		wait = 0
		return 0
	else:
		wait += 1
		if wait >= patience:
			print("Early_stopping triggered")
			return 1

def is_correct(output, label):
	target = [1, 0] if label == "B" else [0, 1]
	output = np.array(output)
	target = np.array(target)
	predicted_class = np.argmax(output)
	target_class = 0 if label == "B" else 1
	if predicted_class == target_class:
		return 1
	else:
		return 0

def cross_entropy(target, output):
	output = np.clip(output, 1e-9, 1.0)
	return -np.sum(target * np.log(output))

def save_weights(layers, filename="weights.npz"):
	weights_dict = {}
	for idx, layer in enumerate(layers):
		weights_matrix = layer.perceptrons
		weights_dict[f"layer_{idx}"] = weights_matrix
	np.savez(filename, **weights_dict)
	return

def load_weights(layers, filename="weights.npz"):
	data = np.load(filename, allow_pickle=True)
	for idx, layer in enumerate(layers):
		key = f"layer_{idx}"
		if key in data:
			weights_matrix = data[key]
			layer.set_weights(weights_matrix)
		else:
			print(f"Warning: {key} not found in {filename}")

class MLP:
	def __init__(self, input_layer=None, output_layer=None, hidden_layers=None, loss=None, learning_rate=None):
		if input_layer is None:
			input_layer = 30
		if output_layer is None:
			output_layer = 2
		if hidden_layers is None:
			hidden_layers = [24, 24]
		if loss is None:
			loss = 'categoricalCrossentropy'
		if learning_rate is None:
			learning_rate = 0.01

		self.loss = loss
		self.learning_rate = learning_rate
		self.layers = []

		#Hidden layers
		prev_layer = None
		for units in hidden_layers:
			layer = Layer(prev_layer, units, "Relu")
			self.layers.append(layer)
			prev_layer = layer

		#Output layer
		self.layers.append(Layer(prev_layer, output_layer, "softmax"))

		self.initialize_weights()

	def train(self, df_train, df_valid, epochs=100, lr=0.035, batch_size=8):
		for epoch in range(epochs):
			load_weights(self.layers)
			total_loss_epoch_train = 0
			total_loss_epoch_validation = 0
			num_batches = 0
			accuracy = 0
			for i in range(0, len(df_train), batch_size):
				batch = df_train.iloc[i:i+batch_size]
				batch_delta = []
				batch_delta.clear()
				for _, row in batch.iterrows():
					output = row[2:].values
					label = row[1]
					for layer in self.layers:
						output = layer.go_forward(output)
					if is_correct(output, label) == 1:
						accuracy +=1
					target = [1, 0] if label == "B" else [0, 1]
					output = np.array(output)
					target = np.array(target)
					total_loss_epoch_train += cross_entropy(target, output)
					batch_delta.append(output - target)
				mean_delta = np.mean(batch_delta, axis=0)
				self.layers[-1].backpropagation(mean_delta, lr) # le -1 permet d'acceder a la derniere couche de la liste
				save_weights(self.layers)
				num_batches += 1
			graph_train_accuracy.append((accuracy / len(df_train)) * 100)
			accuracy = 0
			for _, row in df_valid.iterrows():
				output = row[2:].values
				label = row[1]
				for layer in self.layers:
					output = layer.go_forward(output)
				target = [1, 0] if label == "B" else [0, 1]
				output = np.array(output)
				target = np.array(target)
				total_loss_epoch_validation += cross_entropy(target, output)
				output = np.array(output)
				predicted_class = np.argmax(output)
				target_class = 0 if label == "B" else 1
				if predicted_class == target_class:
					accuracy += 1
			graph_losses_valid.append(total_loss_epoch_validation / len(df_valid))
			graph_losses_train.append(total_loss_epoch_train / len(df_train))
			graph_validation_accuracy.append((accuracy / len(df_valid)) * 100)
			graph_epoch.append(epoch)
			print(f"Epoch {epoch+1} - loss: {total_loss_epoch_train / len(df_train)} - val_loss:{total_loss_epoch_validation / len(df_valid):.4f} - Accuracy: {accuracy / len(df_valid) * 100:.2f}%")
			if early_stopping(total_loss_epoch_validation / len(df_valid)) == 1:
				break
		draw_loss(graph_epoch, graph_losses_train, graph_losses_valid)
		draw_accuracy(graph_epoch, graph_train_accuracy, graph_validation_accuracy)

	def predict(self, df):
		pass
		# load_weights(self.layers)
		# for _, row in df.iterrows():
		# 	output = row[2:].values
		# 	for layer in self.layers:
		# 		output = layer.go_forward(output)
		# 	output = np.array(output)
		
		# print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

	def initialize_weights(self, filename="weights.npz"):
		weights_dict = {}
		for i, layer in enumerate(self.layers):
			input_size = layer.input_size
			size = layer.size
			weights_matrix = np.random.randn(size, input_size + 1) * 0.1
			layer.set_weights(weights_matrix)
			weights_dict[f"layer_{i}"] = weights_matrix
		np.savez(filename, **weights_dict)
