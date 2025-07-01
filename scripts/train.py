from models.mlp import MLP
import pandas as pd

def train_model():
	df_train = pd.read_csv("data/processed/train.csv", header=None)
	df_valid = pd.read_csv("data/processed/valid.csv", header=None)

	mlp = MLP(30, 2, [24, 24])
	mlp.train(df_train, df_valid)