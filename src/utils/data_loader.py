import pandas as pd
import pickle


def load_data(data_path, is_train = True):

	# read the preprocessed data
	data = pd.read_csv(data_path)

	if is_train:
		# drop the targert and categorical columns
		X = data.drop(['pm2_5', 'id', 'site_id', 'city', 'country', 'date'], axis = 1)
		# target column
		y = data['pm2_5']
		return X, y
	else:
		# drop categorical columns
		X = data.drop(['id', 'site_id', 'city', 'country', 'date'], axis = 1)
		# target column
		y = None
		return X