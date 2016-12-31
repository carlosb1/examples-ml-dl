.PHONY: clean data visualize requirements
#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROJECT_NAME = shelf_life
PYTHON_INTERPRETER = python
#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
requirements: test_environment
	pip install -r requirements.txt

## Make Dataset
data: 
	$(PYTHON_INTERPRETER) src/data/make_dataset.py

visualize:
	$(PYTHON_INTERPRETER) src/visualization/visualize.py

extract_features:
	$(PYTHON_INTERPRETER) src/features/build_features.py

## Delete all compiled Python files
clean:
	find . -name "*.pyc" -exec rm {} \;

## Test python environment is setup correctly
test_environment:
	$(PYTHON_INTERPRETER) test_environment.py


