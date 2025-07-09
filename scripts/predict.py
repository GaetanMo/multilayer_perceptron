import pickle
import pandas as pd

def normalize(df):
	cols_to_normalize = df.columns[2:]
	df = df.copy()
	mean = df[cols_to_normalize].mean()
	std = df[cols_to_normalize].std().replace(0, 1e-8)
	df[cols_to_normalize] = (df[cols_to_normalize] - mean) / std
	return df

def predict_model(data_path):
	df = pd.read_csv(data_path, header=None)
	df = normalize(df)
	try:
		with open('model.pkl', 'rb') as f:
			mlp = pickle.load(f)
	except:
		print("File .pkl not found !")
		return
	mlp.predict(df)