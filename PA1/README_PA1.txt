**** Before training and testing, run PrepareData.py to clean the data. 
**** judge.csv and dataset.csv must be prepared separately
**** (cleaned files used for training/prediction are included with submission)

	Cleaning dataset.csv, change the following lines:
		7: dataset_file = 'dataset.csv'
		8: dataset_clean_file = 'dataset_clean.csv
		9: has_exited = True

	Cleaning judge.csv, chage the following lines:
		7: dataset_file = 'judge.csv'
		8: dataset_clean_file = 'judge_clean.csv'
		9: has_exited = False

**** Model training done in training-code.py
     Model saved in model.json
     Model weights saved in model.h5

**** Predictions done in test-code.py
     Predictions made in test-code.py saved in judge-pred.csv