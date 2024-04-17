import yaml
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import os


class LightGBMModel:
	def __init__(self, config_path):
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

	def train(self, X_train, y_train):
		model_params = self.config['lightgbm_model']['parameters']
		train_parameters = self.config['lightgbm_model']['train_params']

		# Splitting the data into training and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 149)

		self.model = lgb.LGBMRegressor(force_col_wise=True, verbosity = 2, early_stopping_round=10 ,**model_params)
		

		if self.config['lightgbm_model']['hyperparameter_tuning']['enable']:
			param_grid = self.config['lightgbm_model']['hyperparameter_tuning']['param_grid']
			n_iter = self.config['lightgbm_model']['hyperparameter_tuning']['n_iter']
			scoring = self.config['lightgbm_model']['hyperparameter_tuning']['scoring']
			random_state = self.config['lightgbm_model']['hyperparameter_tuning']['random_state']

			kf = KFold(n_splits = 10, shuffle = True, random_state = random_state)

			self.model = RandomizedSearchCV(
				estimator=self.model,
				param_distributions =  param_grid,
				n_iter = n_iter,
				scoring = scoring,
				cv = kf,
				random_state = random_state,
				verbose = 2,
				
			)

		fit_params = {
			# 'early_stopping_rounds': train_parameters['early_stopping_rounds'],
			'eval_set': [(X_val, y_val)],
			# 'verbose': train_parameters['verbose']
		}

		self.model.fit(X_train, y_train,  eval_set= [(X_val, y_val)])

	def save_model(self):
		directory = self.config['lightgbm_model']['model_saving']['directory']
		filename = self.config['lightgbm_model']['model_saving']['filename']

		os.makedirs(directory, exist_ok=True)
		pickle.dump(self.model, open(f"{directory}/{filename}", 'wb'))