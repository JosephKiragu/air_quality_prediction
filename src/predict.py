import pandas as pd
import pickle
from utils.data_loader import load_data
from sklearn.metrics import mean_squared_error

def predict(model_path, test_data_path):
	with open(model_path, 'rb') as file:
		model = pickle.load(file)

	X_test= load_data(test_data_path, is_train = False)
	predictions = model.predict(X_test)
	# rmse = mean_squared_error(_, predictions, squared=False)
	# print(f"Test RMSE: {rmse:.4f}")

	return predictions

if __name__=='__main__':
	model_path = 'outputs/models/tuned_xgboost_model.pkl'
	# model_path = 'outputs/models/tuned_lightgbm_model.pkl'
	test_data_path = 'data/processed/Test_data_engineered.csv'

	predictions = predict(model_path, test_data_path)
	output_path = 'outputs/predictions/xgboost_predictions.csv'
	# output_path = 'outputs/predictions/lightgbm_predictions.csv'

	pd.DataFrame({'pm2_5': predictions}).to_csv(output_path, index=False)

	print(f"Predictions saved to {output_path}")
