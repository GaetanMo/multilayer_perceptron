import matplotlib.pyplot as plt

def draw_loss(epoch, loss_train, loss_valid):
	plt.figure()
	plt.plot(epoch, loss_train, marker='o', color='blue', label='Train loss', markersize=1)
	plt.plot(epoch, loss_valid, marker='o', color='orange', label='Validation loss', markersize=1)
	plt.title('Loss evolution')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.grid(True)
	plt.legend()
	plt.savefig('loss_vs_epoch.png')
	plt.close()
	return

def draw_accuracy(epoch, train_accuracy, validation_accuracy):
	plt.figure()
	plt.plot(epoch, train_accuracy, marker='o', color='blue', label='Training accuracy', markersize=1)
	plt.plot(epoch, validation_accuracy, marker='o', color='orange', label='Validation accuracy', markersize=1)
	plt.title('Learning Curves')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig('loss_vs_accuracy.png')
	plt.close()
	return