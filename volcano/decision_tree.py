import numpy as np
import random
import graphviz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os.path

from manage_data import *
from copy import deepcopy
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree._tree import TREE_LEAF


def test_accuracy(x_test, y_test, clf_obj):
	y_predict = clf_obj.predict(x_test)
	return accuracy_score(y_test, y_predict)*100

def test_confusion(x_test, y_test, clf_obj):
	y_predict = clf_obj.predict(x_test)
	return confusion_matrix(y_test, y_predict)

def post_prune(clf, x_prune, y_prune, index=0):
	#Post prune tree while accuracy improves on test set
	if clf.tree_.children_left[index] != TREE_LEAF:
		clf = post_prune(clf, x_prune, y_prune, index=clf.tree_.children_left[index])
	if clf.tree_.children_right[index] != TREE_LEAF:
		clf = post_prune(clf, x_prune, y_prune, index=clf.tree_.children_right[index])

	new_clf = deepcopy(clf)
	new_clf.tree_.children_left[index] = TREE_LEAF
	new_clf.tree_.children_right[index] = TREE_LEAF
	if test_accuracy(x_prune, y_prune, new_clf) >= test_accuracy(x_prune, y_prune, clf):
		clf = new_clf
	return clf


def make_graph(clf_obj, name):
	dot_data = tree.export_graphviz(clf_obj, filled=True, rounded=True, special_characters=True, out_file=None)
	graph = graphviz.Source(dot_data)
	graph.format = 'png'
	graph.render("out_data/" + name)


def plot_learning_curve(x_train, y_train, x_test, y_test, x_prune, y_prune):
	test_points = 10
	trials = 5
	tot_list = [0 for i in range(test_points + 1)]
	tot_list_pre = [0 for i in range(test_points + 1)]
	tot_list_post = [0 for i in range(test_points + 1)]
	rng = [i for i in range(1, int(len(x_test)*0.75), int(len(x_test)*0.75/test_points - 1))]


	for j in range(trials):
		count = 0
		for i in rng:
			dtree_classifier_pre = tree.DecisionTreeClassifier(max_depth=12,random_state=0)
			clf = dtree_classifier_pre.fit(x_train[:i], y_train[:i])
			tot_list_pre[count] += test_accuracy(x_test, y_test, clf)

			dtree_classifier = tree.DecisionTreeClassifier(random_state=0)
			clf = dtree_classifier.fit(x_train[:i], y_train[:i])
			tot_list[count] += test_accuracy(x_test, y_test, clf)

			clf = post_prune(clf, x_prune, y_prune)
			tot_list_post[count] += test_accuracy(x_test, y_test, clf)

			count += 1

	tot_list = [i/trials for i in tot_list]
	tot_list_pre = [i/trials for i in tot_list_pre]
	tot_list_post = [i/trials for i in tot_list_post]

	plt.plot(rng, tot_list, color= '#9CBA7F', linewidth=2)
	plt.plot(rng, tot_list_pre, color= '#8A2BE2', linewidth=2)
	plt.plot(rng, tot_list_post, color= '#40e0d0', linewidth=2)
	plt.title('Decision Tree Learning Curve (Post/No Prune)')
	plt.legend(["No Prune", "Pre Prune", "Post Prune"], loc='lower right')
	plt.xlabel('# of Training Points')
	plt.ylabel('Accuracy %')
	plt.show()
	plt.savefig("out_data/d_tree_learning_curve")


def main():
	if not os.path.isfile("train_test_data.pkl"):
		load_data()

	x_train, y_train, x_test, y_test, x_prune, y_prune= read_data()

	dtree_classifier_pre = tree.DecisionTreeClassifier(max_depth=12, random_state=0)
	clf = dtree_classifier_pre.fit(x_train, y_train)
	print(test_confusion(x_test, y_test, clf))
	make_graph(clf, "pre_prune")

	dtree_classifier = tree.DecisionTreeClassifier(random_state=0)
	clf = dtree_classifier.fit(x_train, y_train)
	print(test_confusion(x_test, y_test, clf))
	make_graph(clf, "no_prune")

	clf = post_prune(clf, x_prune, y_prune)
	print(test_confusion(x_test, y_test, clf))
	make_graph(clf, "post_prune")

	# plot_learning_curve(x_train, y_train, x_test, y_test, x_prune, y_prune)


if __name__=="__main__":
	main()

#----------------------------------------------------------------------------------------------------------------------------------------------------
#SOURCES
#https://stackoverflow.com/questions/49428469/pruning-decision-trees
#https://scikit-learn.org/stable/modules/tree.html
#https://heartbeat.fritz.ai/introduction-to-decision-tree-learning-cd604f85e236
#https://www.geeksforgeeks.org/decision-tree-implementation-python/
#https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence