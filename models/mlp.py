from .layer import Layer
from .graph.drawing import draw_loss
from .graph.drawing import draw_accuracy
import numpy as np
import pandas as pd

graph_epoch = []
graph_losses_train = []
graph_losses_valid = []
graph_train_accuracy = []
graph_validation_accuracy = []
best_loss_validation = float('inf')
patience = 15
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
	output = np.clip(output, 1e-9, 1.0) #avoid 0
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
	def __init__(self, input_layer=None, output_layer=None,  hidden_layers=None, epochs=None, batch_size=None, learning_rate=None, loss=None):
		if input_layer is None:
			input_layer = 30
		if output_layer is None:
			output_layer = 2
		if hidden_layers is None:
			hidden_layers = [24, 24]
		if loss is None:
			loss = 'categoricalCrossentropy'
		if learning_rate is None:
			learning_rate = 0.001
		if epochs is None:
			epochs = 50
		if batch_size is None:
			batch_size = 8
		self.loss = loss
		self.lr = learning_rate
		self.layers = []
		self.epochs = epochs
		self.batch_size = batch_size

		#Hidden layers
		prev_layer = None
		for units in hidden_layers:
			layer = Layer(prev_layer, units, "Relu")
			self.layers.append(layer)
			prev_layer = layer

		#Output layer
		self.layers.append(Layer(prev_layer, output_layer, "softmax"))

		self.initialize_weights()

	def train(self, df_train, df_valid):
		for epoch in range(self.epochs):
			total_loss_epoch_train = 0
			total_loss_epoch_validation = 0
			num_batches = 0
			accuracy = 0
			for i in range(0, len(df_train), self.batch_size): # Training
				batch = df_train.iloc[i:i+self.batch_size]
				batch_delta = []
				batch_delta.clear()
				for _, row in batch.iterrows():
					features = row[2:].values
					label = row[1]
					output = features
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
				self.layers[-1].backpropagation(mean_delta, self.lr)
				num_batches += 1
			graph_train_accuracy.append((accuracy / len(df_train)) * 100)
			accuracy = 0
			for _, row in df_valid.iterrows(): # Validation
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
		for index, row in df.iterrows():
			output = row[2:].values
			for layer in self.layers:
				output = layer.go_forward(output)
			output = np.array(output)
			predicted_class = np.argmax(output)
			if predicted_class == 0:
				predicted_class = "B"
			else:
				predicted_class = "M"
			print(f"Prediction {index + 1} : {predicted_class}")

	def initialize_weights(self):
		for i, layer in enumerate(self.layers):
			input_size = layer.input_size
			size = layer.size
			weights_matrix = np.random.randn(size, input_size + 1) * 0.1
			layer.set_weights(weights_matrix)
