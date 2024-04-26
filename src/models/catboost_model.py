import yaml
import catboost as cb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import optuna
import numpy as np



class CatBoostModel:
	def __init__(self, config_path):
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

	def objective(self, trial, X_train, y_train, kf):
		param = {
			'bootstrap_type': 'Bernoulli',
			'verbose': False,
			'n_estimators': trial.suggest_int('n_estimators', 1100, 1500),
			'od_wait': 50,
			'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
			'reg_lambda': trial.suggest_int('reg_lambda', 55, 70),
			'subsample': trial.suggest_float('subsample', 0.8, 1),
			'random_strength': trial.suggest_int('random_strength', 11, 20),
			'max_depth': trial.suggest_int('max_depth', 5, 10),
			'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 3),
			'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations', 7, 9),
			'loss_function': 'RMSE',
			'random_seed': 149
		}

		rmses = []
		for train_index, val_index in kf.split(X_train):
			X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
			y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

			model = cb.CatBoostRegressor(**param)
			model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose = False)
			preds = model.predict(X_val)
			rmse = mean_squared_error(y_val, preds, squared=False)
			rmses.append(rmse)



		return np.mean(rmses)

	
	def train(self, X_train, y_train):
		kf = KFold(n_splits=self.config['catboost_model']['hyperparameter_tuning']['cv'], shuffle=True, random_state = 149)

		if self.config['catboost_model']['hyperparameter_tuning']['enable']:
			print(f"Tuning the catboost model")
			study = optuna.create_study(direction = 'minimize')
			study.optimize(lambda trial: self.objective(trial, X_train, y_train, kf), n_trials = 70)
			best_params = study.best_params
			best_params['random_seed'] = 149
		else:
			print(f"Training model on Tuned hyperparameters")
			best_params = self.config['catboost_model']['parameters']

		# Training final_model on the full training set with best params
		self.model = cb.CatBoostRegressor(**best_params)
		self.model.fit(X_train, y_train, early_stopping_rounds=50, verbose = True)


	def save_model(self):
		directory = self.config['catboost_model']['model_saving']['directory']
		filename = self.config['catboost_model']['model_saving']['filename']
		pickle.dump(self.model, open(f"{directory}/{filename}", 'wb'))



