import argparse
from scripts.split import split_data
from scripts.train import train_model
from scripts.predict import predict_model

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="MLP: split / train / predict")
	subparsers = parser.add_subparsers(dest="action", required=True)

	parser_split = subparsers.add_parser("split", help="Split the dataset")
    
	parser_train = subparsers.add_parser("train", help="Train the model")
	parser_train.add_argument('--layer', nargs='+', type=int, required=True, help="Hidden layer sizes, e.g. --layer 24 24 24")
	parser_train.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
	parser_train.add_argument('--batch_size', type=int, required=True, help="Batch size")
	parser_train.add_argument('--learning_rate', type=float, required=True, help="Learning rate")

	parser_predict = subparsers.add_parser("predict", help="Predict from a trained model")
	parser_predict.add_argument('--data_path', type=str, help="Path to CSV data file")

	args = parser.parse_args()

	if args.action == "split":
		split_data()

	elif args.action == "train":
		train_model(
			layers=args.layer,
			epochs=args.epochs,
			batch_size=args.batch_size,
			lr=args.learning_rate
		)

	elif args.action == "predict":
		predict_model(args.data_path)

