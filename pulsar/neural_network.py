import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

import numpy as np 
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from utils import *
from math import sqrt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class f1_calc(Callback):
	def on_train_begin(self, logs={}):
		self.f1s = []

	def on_epoch_end(self, epoch, logs={}):
		y_predict = np.array(self.model.predict(self.validation_data[0]))
		y_predict = np.argmax(y_predict, axis=1)
		y_true = self.validation_data[1]
		f1 = f1_score(y_true, y_predict)
		print("F1 score: {}".format(f1))
		self.f1s.append(f1)



def train_model(x_train, y_train, x_test, y_test, history, iterats=1):
	model = Sequential()
	model.add(Dense(units=8, input_dim=8))
	model.add(Dense(units=8))
	model.add(LeakyReLU(alpha = 0.1))
	model.add(Dense(units=8))
	model.add(LeakyReLU(alpha = 0.1))
	model.add(Dense(units=2, activation='softmax'))

	model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=iterats, validation_data=(x_test, y_test), callbacks=[history])

	return model

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

def plot_learning_curves(x_train, y_train, x_test, y_test, trials=10, epochs=100):
	f1 = [0 for i in range(epochs)]

	for i in range(trials):
		history = f1_calc()
		model = train_model(x_train, y_train, x_test, y_test, history, epochs)
		f1 = [sum(x) for x in zip(f1, history.f1s)]

	f1 = [x/trials for x in f1]

	test_metrics_nn(x_test, y_test, model)

	plt.plot(f1, color='#9CBA7F', linewidth=2)
	plt.xlabel('Epoch')
	plt.ylabel('F-measure')
	plt.title('Model F-measure')
	plt.grid(True)
	plt.legend(['Test set f-measure'], loc='lower right')
	plt.savefig("out_data/nn_learning_curve")
	plt.show()

	return model


def main():
	print('\n\n____________________________________')
	print('Neural Network - Pulsar Stars')
	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test= read_data()

	history = f1_calc()
	model = train_model(x_train, y_train, x_test, y_test, history)
	test_metrics_nn(x_test, y_test, model)

	# model = plot_learning_curves(x_train, y_train, x_test, y_test)



if __name__ == '__main__':
	main()
