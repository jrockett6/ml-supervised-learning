import pickle
import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from PIL import Image


def get_picture_resized(df_data, row_loc, pic_res):
	data = (np.array((df_data.iloc[row_loc]), dtype=np.int8))
	data.resize(110, 110)
	img = Image.fromarray(data)
	img = img.resize((pic_res, pic_res), resample=Image.BILINEAR)
	new_arr = np.array(img)
	new_arr.resize((1, pic_res*pic_res))

	return new_arr

def resize_images(df_data, pic_res):
	new_arr = np.array([])

	for i in range(len(df_data)):
		new_row = get_picture_resized(df_data, i, pic_res)

		if not new_arr.size:
			new_arr = new_row
		else:
			new_arr = np.vstack((new_arr, new_row))

	return new_arr


def load_data(nn=False):
	if not nn:
		image_res = 55
	else:
		image_res = 55

	train_data_labels = read_csv("data/train_labels.csv")
	train_data_images = read_csv("data/train_images.csv", header=None)

	test_data_labels = read_csv("data/test_labels.csv")
	test_data_images = read_csv("data/test_images.csv", header=None)

	y_train = train_data_labels['Volcano?']
	y_test = test_data_labels['Volcano?']

	x_train = resize_images(train_data_images, image_res)
	x_test = resize_images(test_data_images, image_res)

	if not nn:
		x_test, x_prune, y_test, y_prune = train_test_split(x_test, y_test, test_size=0.5, random_state=0)

		train_test_data = {'x_train': x_train, 
						   'y_train': y_train,
						   'x_test': x_test, 
						   'y_test': y_test, 
						   'x_prune': x_prune, 
						   'y_prune': y_prune}
		with open('train_test_data.pkl', 'wb') as output:
			pickle.dump(train_test_data, output, -1)

	else:
		train_test_data = {'x_train': x_train, 
						   'y_train': y_train,
						   'x_test': x_test, 
						   'y_test': y_test}	
		with open('train_test_data_nn.pkl', 'wb') as output:
			pickle.dump(train_test_data, output, -1)	   

	print('Training data for images loaded successfully')


def read_data(nn=False):
	if not nn:
		with open('train_test_data.pkl', 'rb') as input:
			train_test_data = pickle.load(input)

		x_train = train_test_data['x_train']
		y_train = train_test_data['y_train']

		x_test = train_test_data['x_test']
		y_test = train_test_data['y_test']

		x_prune = train_test_data['x_prune']
		y_prune= train_test_data['y_prune']

		print('Training data read successfully')

		return x_train, y_train, x_test, y_test, x_prune, y_prune
	else:
		with open('train_test_data_nn.pkl', 'rb') as input:
			train_test_data = pickle.load(input)

		x_train = train_test_data['x_train']
		y_train = train_test_data['y_train']

		x_test = train_test_data['x_test']
		y_test = train_test_data['y_test']

		print('Training data read successfully')

		return x_train, y_train, x_test, y_test



def view_picture(df_data, row_loc, pic_res=110):
	data = (np.array((df_data.iloc[row_loc]), dtype=np.int8))
	data.resize(110, 110)
	img = Image.fromarray(data)
	img = img.resize((pic_res, pic_res), resample=Image.BILINEAR)
	img.save("out_data/volcano_res_ " + str(pic_res) + ".png")
	img.show()
