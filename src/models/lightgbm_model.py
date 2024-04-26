import yaml
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle
import os
import optuna


class LightGBMModel:
	def __init__(self, config_path):
		with open(config_path, 'r') as file:
			self.config = yaml.safe_load(file)

	# def objective(self, trial, X_train, y_train, X_val, y_val):
	# 	param_grid = self.config['lightgbm_model']['hyperparameter_tuning']['param_grid']
	# 	params = {
    #         'objective': 'regression',
    #         'metric': 'rmse',
    #         'verbosity': -1,
    #         'boosting_type': 'gbdt',
    #         'num_leaves': trial.suggest_int('num_leaves', 18, 50),
    #         'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
    #         'n_estimators': trial.suggest_int('n_estimators', 5000, 7000),
    #         'subsample': trial.suggest_float('subsample', 0.6, 0.9),
    #         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
    #         'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.5),
    #         'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
    #         'cat_smooth': trial.suggest_int('cat_smooth', 12, 20),
    #         'min_child_samples': trial.suggest_int('min_child_samples', 15, 30),
    #         'max_depth': trial.suggest_int('max_depth', 1, 4),
    #         'random_state': 149
    #     }
	# 	model = lgb.LGBMRegressor(**params)
	# 	model.fit(X_train, y_train, eval_set = [(X_val, y_val)])
	# 	y_pred = model.predict(X_val)
	# 	rmse = mean_squared_error(y_val, y_pred, squared = False)
	# 	return rmse
	

	

	def train(self, X_train, y_train) :
		# train_parameters = self.config['lightgbm_model']['train_params']
		model_params = self.config['lightgbm_model']['parameters']


		if self.config['lightgbm_model']['hyperparameter_tuning']['enable']:
			param_grid = self.config['lightgbm_model']['hyperparameter_tuning']['param_grid']
			scoring = self.config['lightgbm_model']['hyperparameter_tuning']['scoring']
			random_state = self.config['lightgbm_model']['hyperparameter_tuning']['random_state']
			verbose = self.config['lightgbm_model']['hyperparameter_tuning']['verbose']
			cv = KFold(n_splits=5, shuffle=True, random_state=model_params.get('random_state', 149))


			model = lgb.LGBMRegressor(**model_params)
			model.set_params(early_stopping_round = 50, verbosity = 0, random_state = random_state)

			grid_search = GridSearchCV(
				estimator = model,
				param_grid = param_grid,
				scoring = scoring,
				cv = cv,
				verbose = verbose
			)

			grid_search.fit(X_train, y_train)
			self.model = grid_search.best_estimator_
		else:
			print(f"Hyperparameter tuning is disabled")




	def save_model(self):
		directory = self.config['lightgbm_model']['model_saving']['directory']
		filename = self.config['lightgbm_model']['model_saving']['filename']

		os.makedirs(directory, exist_ok=True)
		pickle.dump(self.model, open(f"{directory}/{filename}", 'wb'))