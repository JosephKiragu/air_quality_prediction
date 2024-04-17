import yaml
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import optuna

class XGBoostModel:
	def __init__(self, config_path):
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

	def objective(self, trial, X_train, y_train, X_val, y_val):
		params = {
			'max_depth': trial.suggest_int('max_depth', 4, 7),
			'min_child_weight': trial.suggest_int('min_child_weight', 3, 6),
			'gamma': trial.suggest_float('gamma', 0, 0.4),
			'subsample': trial.suggest_float('subsample', 0.5, 0.8),
			'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
			'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log = True),
			'n_estimators': trial.suggest_int('n_estimators', 1300, 3000),
			'reg_alpha': trial.suggest_int('reg_alpha', 50, 100),
			'reg_lambda': trial.suggest_int('reg_lambda', 10, 30)

		}

		model = xgb.XGBRegressor(**params)
		model.set_params(early_stopping_rounds = 50)
		model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = 0)
		y_pred = model.predict(X_val)
		rmse = mean_squared_error(y_val, y_pred, squared=False)
		return rmse



	def train(self, X_train, y_train):
		model_params = self.config['xgboost_model']['parameters']
		train_params = self.config['xgboost_model']['train_params']

		# Splitting the data into training and validation sets
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state = 149)

		if self.config['xgboost_model']['hyperparameter_tuning']['enable']:
			study = optuna.create_study(direction = 'minimize')
			study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials = 50)

			best_params = study.best_params
			print(f"Best parameters: {best_params}")
			model_params.update(best_params)

		self.model = xgb.XGBRegressor(**model_params)

		self.model.fit(X_train, y_train, eval_set = [(X_val, y_val)], **train_params)


	def save_model(self):
		directory = self.config['xgboost_model']['model_saving']['directory']
		filename = self.config['xgboost_model']['model_saving']['filename']
		pickle.dump(self.model, open(f"{directory}/{filename}", 'wb'))


		
	

