import yaml
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.lightgbm_model import LightGBMModel
from utils.data_loader import load_data
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

def generate_base_model_predictions(models, X_val) :
	predictions = {}
	for model_name, model in models.items():
		predictions[model_name] = model.predict(X_val)
	return predictions


def main():
	# load configuration files
	with open("outputs/models/tuned_catboost_model.pkl", "rb") as file:
		catboost_model = pickle.load(file)
	with open("outputs/models/tuned_xgboost_model.pkl", "rb") as file:
		xgboost_model = pickle.load(file)
	with open("outputs/models/tuned_lightgbm_model.pkl", "rb") as file:
		lightgbm_model = pickle.load(file)

	models = {
			"xgboost" : xgboost_model,
			"catboost" : catboost_model,
			"lightgbm" : lightgbm_model
			}
	train_data_path = "data/processed/Train_data_engineered.csv"

	X, y = load_data(train_data_path, is_train = True)
	
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2 , random_state=149)

	# saving the y_val in validation labels
	np.save('outputs/validation_labels/y_val.npy', y_val)
 

	base_model_predictions = generate_base_model_predictions(models, X_val)

	# save predictions
	np.save('outputs/predictions/base_model_predictions/base_model_predictions.npy', base_model_predictions)


if __name__ == '__main__':
	main()


	

	


