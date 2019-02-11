import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import os
from sklearn.svm import SVC
from utils import *


def plot_learning_curve(x_train, y_train, x_test, y_test, test_points=15, trials=7):
	tot_list = [0 for i in range(test_points + 1)]
	tot_list_train = [0 for i in range(test_points + 1)]
	rng = [i for i in range(50, int(len(x_train)*0.75), int(len(y_train)*0.75/test_points - 1))]

	for j in range(trials):
		count = 0
		print(j)
		for i in rng:
			clf = SVC(kernel='rbf', gamma='auto')
			clf.fit(x_train[:i], y_train[:i])

			tot_list[count] += test_metrics(x_test, y_test, clf)
			tot_list_train[count] += test_metrics(x_train, y_train, clf)

			count += 1

	tot_list = [i/trials for i in tot_list]
	tot_list_train = [i/trials for i in tot_list_train]

	test_metrics(x_test, y_test, clf, do_print=True)

	plt.plot(rng, tot_list_train[:-1], color='#8A2BE2', linewidth=2)
	plt.plot(rng, tot_list[:-1], color='#9CBA7F', linewidth=2)
	plt.title('SVM Learning Curve - RBF')
	plt.xlabel('# of Training Points')
	plt.ylabel('F-measure')
	plt.legend(['Train', 'Test'], loc='lower right')
	plt.grid(True)
	plt.savefig("out_data/svm_learning_curve")
	plt.show()


def main():
	print('\n\n____________________________________')
	print('Support Vector Machine - Pulsar Stars')
	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test = read_data()

	clf = SVC(gamma='auto')
	clf.fit(x_train, y_train)
	test_metrics(x_test, y_test, clf, do_print=True)

	# plot_learning_curve(x_train, y_train, x_test, y_test)




if __name__=="__main__":
	main()