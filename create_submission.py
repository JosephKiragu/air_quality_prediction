import pandas as pd
import os

def create_submission(test_data_path, predictions, output_path):
	test_data = pd.read_csv(test_data_path)
	submission = pd.DataFrame({'id': test_data['id'], 'pm2_5': predictions})

	os.makedirs(os.path.dirname(output_path), exist_ok = True)
	submission.to_csv(output_path, index = False)
	print(f"Submission file saved to {output_path}")

if __name__ == '__main__':
	test_data_path = 'data/processed/Test_data_engineered.csv'
	predictions_path = 'outputs/predictions/xgboost_predictions.csv'
	output_path = 'outputs/submissions/xgboost_submission.csv'
	# predictions_path = 'outputs/predictions/lightgbm_predictions.csv'
	# output_path = 'outputs/submissions/lightgbm_submission.csv'

	predictions = pd.read_csv(predictions_path)['pm2_5'].values
	create_submission(test_data_path, predictions, output_path)
