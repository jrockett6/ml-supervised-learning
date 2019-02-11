TO RUN THE CODE
_____________________________
prerequisites:

All models were implemented using the most updated versions of the following packages. Exact versions are listed. However, any recent versions of the following packages will likely work.

Python 3.5.6
Numpy 1.15.2
Scikit-learn 0.20
Pandas 0.23.4
Keras 2.2.2
Pillow 5.2.0
Matplotlib 3.0.0
Graphviz 2.40.1 (conda install requires python-graphviz)

_____________________________
STEPS

1:	Clone or download code repository from https://github.com/jrrockett/ML-Supervised-Learning
2:	In desired dataset folder (volcano or pulsar) create two folders - "data" and "out_data"
3:	Download desired data from appropriate location, unzip any .tar.gz files and put all .csv files directly into respective created "data" folder:
		Pulsar: https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star
		Volcano: https://www.kaggle.com/fmena14/volcanoesvenus
4:	In desired folder, run "run_models.py" script with listed prerequistes installed
		NOTE: Learning curve functions are commented for reasonable run times. If you'd like to see the learning curves plotted, uncomment the line in each file's main function.
