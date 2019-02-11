import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os

from utils import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


def test_accuracy(x_test, y_test, clf_obj):
	y_predict = clf_obj.predict(x_test)
	return accuracy_score(y_test, y_predict)*100

def plot_learning_curve(x_train, y_train, x_test, y_test, test_points=10, trials=10):
	tot_list = [0 for i in range(test_points + 1)]
	tot_list_train = [0 for i in range(test_points + 1)]
	rng = [i for i in range(2, int(len(x_train)), int(len(x_train)/test_points - 1))]

	for j in range(trials):
		count = 0
		print(j)
		for i in rng:
			dtree_classifier = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
			clf = AdaBoostClassifier(n_estimators=100, base_estimator=dtree_classifier,learning_rate=1)
			clf.fit(x_train[:i], y_train[:i])

			tot_list[count] += test_accuracy(x_test, y_test, clf)
			tot_list_train[count] += test_accuracy(x_train, y_train, clf)

			count += 1

	tot_list = [i/trials for i in tot_list]
	tot_list_train = [i/trials for i in tot_list_train]

	test_metrics(x_test, y_test, clf)

	plt.plot(rng, tot_list_train, color='#8A2BE2', linewidth=2)
	plt.plot(rng, tot_list, color='#9CBA7F', linewidth=2)
	plt.title('Boosted Dtree Learning Curve')
	plt.xlabel('# of Training Points')
	plt.legend(['Train', 'Test'], loc='lower right')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.show()
	plt.savefig("out_data/boosted_learning_curve")

def main():
	print('\n\n_______________________________________')
	print('Boosted Dtree - Volcanoes on Venus')

	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test = read_data()

	dtree_classifier = tree.DecisionTreeClassifier(max_depth=2, random_state=0)
	clf = AdaBoostClassifier(n_estimators=500, base_estimator=dtree_classifier,learning_rate=1)
	clf.fit(x_train, y_train)
	test_metrics(x_test, y_test, clf)

	# plot_learning_curve(x_train, y_train, x_test, y_test, test_points=15, trials=1)


if __name__ == "__main__":
	main()
