import pickle
import pandas as pd

def load_model(file_path):
	with open(file_path, 'rb') as file:
		model = pickle.load(file)
	return model

def get_model_parameters(model):
	parameters = model.get_params()
	return parameters

model_list = ['tuned_catboost_model.pkl', 'tuned_lightgbm_model.pkl', 'tuned_xgboost_model.pkl']

# retrieve parameters
for models in model_list:
	model_path = f'outputs/models/{models}'
	model = load_model(model_path)
	model_parameters = get_model_parameters(model)
	# save parameters to in a txt file in tabular format
	with open('outputs/model weights/trained_model_parameters.txt', 'a') as file:
		file.write(f"Model parameters for {models}: {model_parameters}")
		# put in next row
		file.write(" \n")
	print(f"Model parameters for {models} : {model_parameters}")
	print(" /n")
	
