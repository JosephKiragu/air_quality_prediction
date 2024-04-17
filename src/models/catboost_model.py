import yaml
import catboost as cb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle


class CatBoostModel:
	def __init__(self, config_path):
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

	def train(self, X_train, y_train):
		model_params = self.config['catboost_model']['parameters']
		train_params = self.config['catboost_model']['train_params']

		self.model = cb.CatBoostRegressor(**model_params)

		if self.config['catboost_model']['hyperparameter_tuning']['enable']:
			param_grid = self.config['catboost_model']['hyperparameter_tuning']['param_grid']
			cv = self.config['catboost_model']['hyperparameter_tuning']['cv']
			scoring = self.config['catboost_model']['hyperparameter_tuning']['scoring']
			verbose = self.config['catboost_model']['hyperparameter_tuning']['verbose']

			grid_search = GridSearchCV(
				estimator=self.model,
				param_grid = param_grid,
				cv = cv,
				scoring = scoring,
				verbose = verbose
			)

			grid_search.fit(X_train, y_train, **train_params)
			self.model = grid_search.best_estimator_

	def save_model(self):
		directory = self.config['catboost_model']['model_saving']['directory']
		filename = self.config['catboost_model']['model_saving']['filename']
		pickle.dump(self.model, open(f"{directory}/{filename}", 'wb'))



