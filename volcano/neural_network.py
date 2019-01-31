import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from manage_data import *



# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


#Create model, Sequential NN
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)), #reformats first layer into single 1d flat array
#     keras.layers.Dense(128, activation=tf.nn.relu), #first layer, 128 nodes
#     keras.layers.Dense(10, activation=tf.nn.softmax) #second layer, 10 node "softmax layer" returns array of 10 
# ])

# #Compile the model
# model.compile(optimizer=tf.train.AdamOptimizer(),
# 			loss='sparse_categorical_crossentropy',
# 			metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs = 5)

# test_loss, test_acc = model.evaluate(test_images, test_labels)

# print('Test accuracy:', test_acc)

# predictions = model.predict(test_images)
	
# pre


def main():
	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test, x_prune, y_prune= read_data()

	print(type(x_test))
	print(type(x_prune))
	print(x_test.shape)


if __name == '__main__':
	main()

#https://www.tensorflow.org/tutorials/keras/basic_classification
