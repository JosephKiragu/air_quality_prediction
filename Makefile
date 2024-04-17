
run-fe:
	python3 src/data_preprocessing/feature_engineer.py
	@echo "Feature Engineering Completed"

train:
	python3 src/train.py
	@echo "Model training completed"

predict:
	python3 src/predict.py
	@echo "Predictions Generated"

submit:
	python3 create_submission.py
	@echo "Submission file generated"

clean-models:
	rm -rf output/models/*
	@echo "Cleaned all trained models"

clean-predictions:
	rm -rf output/predictions/*
	@echo "Cleaned all predictions"

clean-processed:
	rm -rf data/processed/*.csv
	@echo "Cleaned all processed data"

clean-submission:
	rm -rf outputs/submissions/*.csv
	@echo "Cleaned all submission files"


clean: clean-models clean-predictions clean-processed clean-Submission
	@echo "Cleaned all sections"

all: run-fe train predict

.PHONY: all run-fe train predict clean clean-models clean-predictions clean-processed