import pandas as pd

def split_data():
	df = pd.read_csv("data/raw/data.csv", header=None)
	df = df.sample(frac=1, random_state=38).reset_index(drop=True)

	n_valid = int(len(df) * 0.1)	
	df_valid = df.iloc[:n_valid].copy()
	df_train = df.iloc[n_valid:].copy()
	df_train.to_csv("data/processed/train.csv", index=False, header=False)
	df_valid.to_csv("data/processed/valid.csv", index=False, header=False)
