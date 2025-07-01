import pandas as pd

def normalize_with_mean_std(df, mean, std):
    cols_to_normalize = df.columns[2:]
    df.loc[:, cols_to_normalize] = (df[cols_to_normalize] - mean) / std
    return df

def normalize(df):
    cols_to_normalize = df.columns[2:]
    mean = df[cols_to_normalize].mean()
    std = df[cols_to_normalize].std()
    df.loc[:, cols_to_normalize] = (df[cols_to_normalize] - mean) / std
    return df, mean, std

def split_data():
	df = pd.read_csv("data/raw/data.csv")

	n_valid = int(len(df) * 0.1)
	df_valid = df.iloc[:n_valid].copy()
	df_train = df.iloc[n_valid:].copy()

	df_train, mean, std = normalize(df_train)
	df_valid = normalize_with_mean_std(df_valid, mean, std)
	
	df_train.to_csv("data/processed/train.csv", index=False)
	df_valid.to_csv("data/processed/valid.csv", index=False)