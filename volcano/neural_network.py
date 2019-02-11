import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_model(x_train, y_train, x_test, y_test, iterats=1):
	model = Sequential()
	model.add(Dense(units=55, input_dim=3025))
	model.add(Dense(units=30))
	model.add(LeakyReLU(alpha=0.1))
	model.add(Dropout(rate=0.2))
	model.add(Dense(units=2, activation='softmax'))


	model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(x_train, y_train, epochs=iterats, validation_data=(x_test, y_test))

	return model, history

def test_metrics_nn(x_test, y_test, model):
	y_predict = model.predict(x_test)
	y_predict = np.argmax(y_predict, axis=1)
	conf_mat = confusion_matrix(y_test, y_predict)

	true_pos = conf_mat[1][1]
	false_pos = conf_mat[0][1]
	false_neg = conf_mat[1][0]
	if (true_pos + false_pos) == 0:
		return 0

	precision = true_pos/(true_pos + false_pos)
	print('Precision: ', str(precision))
	recall = true_pos/(true_pos + false_neg)
	print('Recall: ', str(recall))
	f1 = 2 * (precision*recall) / (precision+recall)
	print('F-measure: ', str(f1))
	print('Accuracy: ', str(accuracy_score(y_test, y_predict)*100))
	print('Confusion matrix:')
	print(conf_mat)
	print()
	return conf_mat

def plot_learning_curves(x_train, y_train, x_test, y_test, trials=100, epochs=100):
	train_acc = [0 for i in range(epochs)]
	test_acc = [0 for i in range(epochs)]

	for i in range(trials):
		model, history = train_model(x_train, y_train, x_test, y_test, epochs)

		train_acc = [sum(x) for x in zip(train_acc, history.history['acc'])]
		test_acc = [sum(x) for x in zip(test_acc, history.history['val_acc'])]

	train_acc = [i/trials for i in train_acc]
	test_acc = [i/trials for i in test_acc]

	test_metrics_nn(x_test, y_test, model)

	plt.plot(train_acc, color='#9CBA7F', linewidth=2)
	plt.plot(test_acc, color='#8A2BE2', linewidth=2)
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.title('Model Accuracy')
	plt.grid(True)
	plt.legend(['Train', 'Test'], loc='lower right')
	plt.savefig("out_data/nn_learning_curve")
	plt.show()

	return model


def main():
	print('\n\n_______________________________________')
	print('Neural Network - Volcanoes on Venus')

	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test= read_data()

	model, history = train_model(x_train, y_train, x_test, y_test, 1)
	test_metrics_nn(x_test, y_test, model)
	# model = plot_learning_curves(x_train, y_train, x_test, y_test, trials=10, epochs=100)


if __name__ == '__main__':
	main()

