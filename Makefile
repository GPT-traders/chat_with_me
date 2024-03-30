.PHONY: all

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROFILE = default
PROJECT_NAME = Chat-with-me
PYTHON_INTERPRETER = python
PYTHON_PATH=$(shell pwd)/

# Operating System
UNAME_S := $(shell uname -s)

#################################################################################
# COMMANDS                                                                      #
#################################################################################

show:
	@echo "\n**********Info***************"
	@echo PROJECT_DIR=${PROJECT_DIR}
	@echo PROJECT_NAME=${PROJECT_NAME}
	@echo DATETIME_STAMP=${DATETIME_STAMP}
	@echo PYTHON_PATH=${PYTHON_PATH}
	@echo "*************************\n"
    
# Delete all compiled Python files
clean: format
	@echo $(filter "^192", $(LOCALIP))
	@echo $(DATETIME_STAMP)
	@echo $(GIT_COMMIT)
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
# Format import statements
isort:
	isort .

# Format all repo files with black
black:
	black .

# Check source formatting (doesn't support pyproject.toml, so config is here)
flake8:
	flake8 \
		--max-line-length 100 \
		--exclude .pytest_cache,.venv,.vscode,.coverage,.gitignore,temp,setup.py,.ipynb_checkpoints,docs

# Format and check all python source code
format: isort black flake8

#start qdrant in local
qdrant-start:
	@echo $("Starting Qdrant db")
	docker run -p 6333:6333 \
    		-v $(pwd)/data:/qdrant/storage \
    		qdrant/qdrant

start-local-app:
	uvicorn app.main:app --reload
	