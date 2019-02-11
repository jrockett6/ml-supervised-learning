import numpy as np
import random
import graphviz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os.path

from utils import *
from copy import deepcopy
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF


def post_prune(clf, x_prune, y_prune, index=0):
	#Post prune tree while accuracy improves on test set
	if clf.tree_.children_left[index] != TREE_LEAF:
		clf = post_prune(clf, x_prune, y_prune, index=clf.tree_.children_left[index])
	if clf.tree_.children_right[index] != TREE_LEAF:
		clf = post_prune(clf, x_prune, y_prune, index=clf.tree_.children_right[index])

	new_clf = deepcopy(clf)
	new_clf.tree_.children_left[index] = TREE_LEAF
	new_clf.tree_.children_right[index] = TREE_LEAF
	if test_metrics(x_prune, y_prune, new_clf) >= test_metrics(x_prune, y_prune, clf):
		clf = new_clf
	return clf


def make_graph(clf_obj, name):
	dot_data = tree.export_graphviz(clf_obj, filled=True, rounded=True, special_characters=True, out_file=None)
	graph = graphviz.Source(dot_data)
	graph.format = 'png'
	graph.render("out_data/" + name)


def plot_learning_curve(x_train, y_train, x_test, y_test, x_prune, y_prune):
	test_points = 15
	trials = 20
	tot_list = [0 for i in range(test_points + 1)]
	tot_list_pre = [0 for i in range(test_points + 1)]
	tot_list_post = [0 for i in range(test_points + 1)]
	tot_list_post_train = [0 for i in range(test_points + 1)]
	rng = [i for i in range(1, int(len(x_test)*0.75), int(len(x_test)*0.75/test_points - 1))]


	for j in range(trials):
		count = 0
		print(j)
		for i in rng:
			dtree_classifier_pre = tree.DecisionTreeClassifier(max_depth=12,random_state=0)
			clf = dtree_classifier_pre.fit(x_train[:i], y_train[:i])
			tot_list_pre[count] += test_metrics(x_test, y_test, clf)

			dtree_classifier = tree.DecisionTreeClassifier(random_state=0)
			clf = dtree_classifier.fit(x_train[:i], y_train[:i])
			tot_list[count] += test_metrics(x_test, y_test, clf)

			clf = post_prune(clf, x_prune, y_prune)
			tot_list_post[count] += test_metrics(x_test, y_test, clf)
			tot_list_post_train[count] += test_metrics(x_train, y_train, clf)

			count += 1

	tot_list = [i/trials for i in tot_list]
	tot_list_pre = [i/trials for i in tot_list_pre]
	tot_list_post = [i/trials for i in tot_list_post]
	tot_list_post_train = [i/trials for i in tot_list_post_train]

	test_metrics(x_test, y_test, clf, do_print=True)
	# print(test_accuracy(x_test, y_test, clf))

	plt.plot(rng, tot_list, color= '#9CBA7F', linewidth=2)
	plt.plot(rng, tot_list_pre, color= '#8A2BE2', linewidth=2)
	plt.plot(rng, tot_list_post, color= '#40e0d0', linewidth=2)
	plt.plot(rng, tot_list_post_train, color= '#DC143C', linewidth=2)
	plt.title('Decision Tree Learning Curve')
	plt.legend(["No Prune (test)", "Pre Prune (test)", "Post Prune (test)", "Post Prune (train)"], loc='lower right')
	plt.xlabel('# of Training Points')
	plt.ylabel('F-measure')
	plt.grid(True)
	plt.show()
	plt.savefig("out_data/d_tree_learning_curve")


def main():
	# data = read_csv("data/pulsar_stars.csv")
	# view_data(data)
	print('\n\n____________________________________')
	print('Decision Tree - Pulsar Stars')

	if not os.path.isfile("train_test_data_dtree.pkl"):
		load_data(dtree=True)

	x_train, y_train, x_test, y_test, x_prune, y_prune = read_data(dtree=True)

	dtree_classifier_pre = tree.DecisionTreeClassifier(max_depth=12, random_state=0)
	clf = dtree_classifier_pre.fit(x_train, y_train)
	print("\n\nPre pruned tree\n- - - - - - - - - - - - - ")
	test_metrics(x_test, y_test, clf, do_print=True)
	# make_graph(clf, "pre_prune")

	dtree_classifier = tree.DecisionTreeClassifier(random_state=0)
	clf = dtree_classifier.fit(x_train, y_train)
	print("\nNo pruned tree\n- - - - - - - - - - - - - ")
	test_metrics(x_test, y_test, clf, do_print=True)
	# make_graph(clf, "no_prune")

	clf = post_prune(clf, x_prune, y_prune)
	print("\nPost pruned tree\n- - - - - - - - - - - - - ")
	test_metrics(x_test, y_test, clf, do_print=True)
	# make_graph(clf, "post_prune")

	# plot_learning_curve(x_train, y_train, x_test, y_test, x_prune, y_prune)


if __name__=="__main__":
	main()
