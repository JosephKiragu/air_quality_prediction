import pandas as pd
import pickle
from sklearn.model_selection import KFold


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
	
# def split_data_and_save(X, y, n_splits = 5, random_state = 149) :
# 	kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
# 	fold_data = list(kf.split(X))

# 	for i, (train_index, val_index) in enumerate(fold_data):
# 		X_train, X_val = X.iloc[train_index], X.iloc[val_index]
# 		y_train, y_val = y.iloc[train_index], y.iloc[val_index]

# 		# save these splits to disk
# 		X_train.to_csv(f'data/split/train_fold_{i}.csv', index = False)
# 		X_val.to_csv(f'data/split/val_fold_{i}.csv', index = False)
# 		y_train.to_csv(f'data/split/train_target_fold_{i}.csv', index = False)
# 		y_val.to_csv(f'data/split/val_target_fold_{i}.csv', index = False)


# def load_split_data(train_file, val_file, train_target_file, val_target_file):
#     X_train = pd.read_csv(train_file)
#     X_val = pd.read_csv(val_file)
#     y_train = pd.read_csv(train_target_file).squeeze()  # Squeeze to convert DataFrame to Series if needed
#     y_val = pd.read_csv(val_target_file).squeeze()

#     return X_train, X_val, y_train, y_val
