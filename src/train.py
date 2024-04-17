from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.catboost_model import CatBoostModel
from utils.data_loader import load_data

def train_xgboost_model(config_path):
	X_train, y_train = load_data('data/processed/Train_data_engineered.csv', is_train = True)

	model = XGBoostModel(config_path)

	# train model
	model.train(X_train, y_train)

	# save trained model
	model.save_model()

def train_lightgbm_model(config_path):
	X_train, y_train = load_data('data/processed/Train_data_engineered.csv', is_train = True)

	model = LightGBMModel(config_path)
	model.train(X_train, y_train)
	model.save_model()

def train_catboost_model(config_path):
	X_train, y_train = load_data('data/processed/Train_data_engineered.csv', is_train = True)
	model = CatBoostModel(config_path)
	model.train(X_train, y_train)
	model.save_model()

if __name__ == '__main__':
	xgboost_config_path = 'configs/base_models/xgboost.yml'
	# lightgbm_config_path = 'configs/base_models/lightgbm.yml'
	# catboost_config_path = 'configs/base_models/catboost.yml'

	train_xgboost_model(xgboost_config_path)
	# train_lightgbm_model(lightgbm_config_path)
	# train_catboost_model(catboost_config_path)
