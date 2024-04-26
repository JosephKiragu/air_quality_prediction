import yaml
import numpy as np

def load_predictions(path):
	return np.load(path, allow_pickle=True).item()

def create_meta_features(base_model_predictions):
	meta_features = []
	for model_name, predictions in base_model_predictions.items():
		meta_features.append(predictions)
	meta_features = np.column_stack(meta_features)
	return meta_features

def main():
	# load model predictions
	base_model_predictions = load_predictions('outputs/predictions/base_model_predictions/base_model_predictions.npy')

	#create meta features
	meta_features = create_meta_features(base_model_predictions)

	# save meta features
	np.save("outputs/meta_features/meta_features.npy", meta_features)

if __name__ == '__main__':
	main()
	
