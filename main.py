import argparse
from scripts.split import split_data
from scripts.train import train_model
from scripts.predict import predict_model

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="MLP: split / train / predict")
	parser.add_argument("action", choices=["split", "train", "predict"],
						help="Choices : split / train / predict")
	args = parser.parse_args()
	if args.action == "split":
		split_data()
	elif args.action == "train":
		train_model()
	elif args.action == "predict":
		predict_model("data/raw/data.csv")
