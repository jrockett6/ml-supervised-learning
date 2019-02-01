import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

import numpy as np 
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from manage_data import *
from math import sqrt

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def prepare_data():
	if not os.path.isfile("train_test_data_nn.pkl"):
		load_data(nn=True)

	x_train, y_train, x_test, y_test= read_data(nn=True)

	x_train = x_train/255
	y_train = np.array(y_train)

	x_test = x_test/255
	y_test = np.array(y_test)

	return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, trials=1):
	model = Sequential()
	model.add(Dense(units=55, input_dim=3025))
	model.add(Dense(units=30))
	model.add(LeakyReLU(alpha = 0.01))
	model.add(Dense(units=2, activation='softmax'))

	model.compile(Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=trials)

	return model

def test_confusion(x_test, y_test, model):
	y_predict = model.predict(x_test)
	y_predict = np.argmax(y_predict, axis=1)
	return confusion_matrix(y_test, y_predict)


def main():

	x_train, y_train, x_test, y_test = prepare_data()

	model = train_model(x_train, y_train, 5)

	print(test_confusion(x_test, y_test, model))

	test_loss, test_acc = model.evaluate(x_test, y_test)

	print('Test accuracy:', test_acc)
	print('Log loss:', test_loss)


if __name__ == '__main__':
	main()

#https://www.tensorflow.org/tutorials/keras/basic_classification
#https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
#https://stackoverflow.com/questions/44624334/tensorflow-how-to-display-confusion-matrix-tensor-as-an-array
