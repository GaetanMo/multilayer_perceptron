# 🧠 Multilayer Perceptron (MLP) - 42 Project

A simple feedforward neural network built entirely from scratch in Python — **without using any machine learning frameworks** (no TensorFlow, PyTorch, or scikit-learn).

> This project was developed as part of the 42 curriculum to deepen understanding of machine learning fundamentals and neural network mechanics.

---

## 🚀 Objectives

- Implement a **multilayer perceptron (MLP)** from the ground up.
- Gain a solid grasp of:
  - Forward propagation
  - Backpropagation
  - Gradient descent
  - Training on labeled datasets
- Handle basic binary and multi-class classification problems.

---

## 🛠️ Features

- Configurable architecture: number of layers, neurons per layer.
- Supported activation function: **ReLU**.
- Loss function: **Cross-Entropy**.
- Training feedback: live monitoring of **loss**, **accuracy**, etc.

---

## 📦 Installation

```bash
git clone https://github.com/your-username/multilayer_perceptron.git

## 🔧 Usage

Run the main script with one of the following actions: `split`, `train`, or `predict`.

### 1. Split the dataset

Split your raw dataset (CSV format) into training and validation sets:

```bash
python main.py split

### 2. Train the model

Train your MLP with customizable architecture and training parameters:

```bash
python main.py train \
  --layer 24 24 24 \
  --epochs 84 \
  --batch_size 8 \
  --learning_rate 0.0314

--layer: Number of neurons in each hidden layer (you can specify multiple).

--epochs: Total number of training iterations over the dataset.

--batch_size: Size of the data batches used during training.

--learning_rate: Step size used by the optimizer.

### 3. Predict using the trained model

Run inference on a dataset using the previously trained model:

```bash
python main.py predict --data_path data/raw/data.csv

✅ Make sure that model.pkl (the trained model) exists before running predictions.
