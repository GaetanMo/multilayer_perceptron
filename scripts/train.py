import pandas as pd
from models.mlp import MLP

def normalize_with_mean_std(df, mean, std):
	cols_to_normalize = df.columns[2:]
	df = df.copy()
	df[cols_to_normalize] = (df[cols_to_normalize] - mean) / std.replace(0, 1e-8)
	return df

def normalize(df):
	cols_to_normalize = df.columns[2:]
	df = df.copy()
	mean = df[cols_to_normalize].mean()
	std = df[cols_to_normalize].std().replace(0, 1e-8)
	df[cols_to_normalize] = (df[cols_to_normalize] - mean) / std
	return df, mean, std

def train_model():
	df_train = pd.read_csv("data/processed/train.csv", header=None)
	df_valid = pd.read_csv("data/processed/valid.csv", header=None)

	df_train, mean, std = normalize(df_train)
	df_valid = normalize_with_mean_std(df_valid, mean, std)

	mlp = MLP(30, 2, [10, 10])
	mlp.train(df_train, df_valid)
