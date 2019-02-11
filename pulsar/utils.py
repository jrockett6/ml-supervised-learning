import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import pickle
import numpy as np
from pandas import read_csv
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def load_data(dtree=False):
	data = read_csv("data/pulsar_stars.csv")
	y_data = np.array(data['target_class'])
	x = data.drop(['target_class'], axis=1)
	x_data = np.array([(x[column] - np.min(x[column]))/(np.max(x[column]) - np.min(x[column])) for column in x]).transpose() #scale between 0/1

	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)

	if dtree:
		x_test, x_prune, y_test, y_prune = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

		train_test_data = {'x_train': x_train, 
						   'y_train': y_train,
						   'x_test': x_test, 
						   'y_test': y_test, 
						   'x_prune': x_prune, 
						   'y_prune': y_prune}
		with open('train_test_data_dtree.pkl', 'wb') as output:
			pickle.dump(train_test_data, output, -1)

	else:
		train_test_data = {'x_train': x_train, 
						   'y_train': y_train,
						   'x_test': x_test, 
						   'y_test': y_test}	

		with open('train_test_data.pkl', 'wb') as output:
			pickle.dump(train_test_data, output, -1)	   

	print('(Training data loaded successfully)\n')


def read_data(dtree=False):
	if dtree:
		with open('train_test_data_dtree.pkl', 'rb') as input:
			train_test_data = pickle.load(input)

		x_train = train_test_data['x_train']
		y_train = train_test_data['y_train']

		x_test = train_test_data['x_test']
		y_test = train_test_data['y_test']

		x_prune = train_test_data['x_prune']
		y_prune= train_test_data['y_prune']

		print('(Training data read successfully)\n')

		return x_train, y_train, x_test, y_test, x_prune, y_prune

	else:	
		with open('train_test_data.pkl', 'rb') as input:
				train_test_data = pickle.load(input)

		x_train = train_test_data['x_train']
		y_train = train_test_data['y_train']

		x_test = train_test_data['x_test']
		y_test = train_test_data['y_test']

		print('(Training data read successfully)\n')

		return x_train, y_train, x_test, y_test


def view_data(data):

	keys = list(data)

	print(keys)

	noise_points = data[data.target_class==0]
	pulsar_points = data[data.target_class==1]

	fig, ax = plt.subplots(2, 2)
	
	ax[0, 1].scatter(noise_points[keys[1]], noise_points[keys[2]],label="RF Noise",color="red",alpha=0.1)
	ax[0, 1].scatter(pulsar_points[keys[1]], pulsar_points[keys[2]],label="Pulsar",color="purple",alpha=0.1)
	ax[0, 1].set_xlabel(keys[1])
	ax[0, 1].set_ylabel(keys[2])
	ax[0, 1].legend()

	ax[0, 0].scatter(noise_points[keys[0]], noise_points[keys[3]],label="RF Noise",color="red",alpha=0.1)
	ax[0, 0].scatter(pulsar_points[keys[0]], pulsar_points[keys[3]],label="Pulsar",color="purple",alpha=0.1)
	ax[0, 0].set_xlabel(keys[0])
	ax[0, 0].set_ylabel(keys[3])

	ax[1, 0].scatter(noise_points[keys[4]], noise_points[keys[7]],label="RF Noise",color="red",alpha=0.1)
	ax[1, 0].scatter(pulsar_points[keys[4]], pulsar_points[keys[7]],label="Pulsar",color="purple",alpha=0.1)
	ax[1, 0].set_xlabel(keys[4])
	ax[1, 0].set_ylabel(keys[7])

	ax[1, 1].scatter(noise_points[keys[5]], noise_points[keys[6]],label="RF Noise",color="red",alpha=0.1)
	ax[1, 1].scatter(pulsar_points[keys[5]], pulsar_points[keys[6]],label="Pulsar",color="purple",alpha=0.1)
	ax[1, 1].set_xlabel(keys[5])
	ax[1, 1].set_ylabel(keys[6])

	fig.suptitle('Pulsar Star vs RF Noise Attributes')
	plt.show()


def test_metrics(x_test, y_test, clf_obj, do_print=False):
	y_predict = clf_obj.predict(x_test)
	conf_mat = confusion_matrix(y_test, y_predict)

	true_pos = conf_mat[1][1]
	false_pos = conf_mat[0][1]
	false_neg = conf_mat[1][0]
	if (true_pos + false_pos) == 0:
		return 0

	precision = true_pos/(true_pos + false_pos)
	recall = true_pos/(true_pos + false_neg)
	f1 = 2 * (precision*recall) / (precision+recall)
	if do_print:
		print('Recall: ', str(recall))
		print('Precision: ', str(precision))
		print('F-measure: ', str(f1))
		print('Accuracy: ', str(accuracy_score(y_test, y_predict)*100))
		print('Confusion matrix:')
		print(conf_mat)
		print()
	return f1
