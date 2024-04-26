import yaml
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
import pickle

def load_meta_features(path):
	return np.load(path)

def load_validation_labels(path) :
	return np.load(path)

def train_meta_model(meta_features, y_val, config) :
	model = LinearRegression(**config['model_params'])

	if config['cross_validation']['cv']:
		cv_results = cross_validate(model, meta_features, y_val, **config['cross_validation'])
		print(f"Cross validation results: {cv_results}")

	model.fit(meta_features, y_val)

	return model, cv_results

def save_model(model, path):
	with open(path, 'wb') as file:
		pickle.dump(model, file)

def main() :
	# load configurations
	with open('configs/meta_model/linear_regression.yml') as file:
		config = yaml.safe_load(file)

	# Load meta-features and validation models
	meta_features = load_meta_features("outputs/meta_features/meta_features.npy")
	y_val = load_validation_labels("outputs/validation_labels/y_val.npy")

	# Train meta model
	model, cv_results = train_meta_model(meta_features, y_val, config)

	# save the trained model
	save_model(model, config['output']['model_path'])

	# save additional outputs
	if config['output']['coefficients']:
		np.save("outputs/meta_model/coefficients.npy", model.coef_)
	if config['output']['intercept']:
		np.save("outputs/meta_model/intercept.npy", model.intercept_)
	if config['output']['cv_results']:
		np.save("outputs/meta_model/cv_reults.npy", cv_results)


if __name__ == '__main__':
	main()
	