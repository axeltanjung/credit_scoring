# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load config from utils file
import utils as utils

CONFIG_DATA = utils.load_config()

def open_data():
    """Import data and dump data"""

    # Import data
    path_data = CONFIG_DATA['raw_data_path']
    data = pd.read_csv(path_data)

    # Create data validation for data shape
    print("Shape of data    :", data.shape)

    # Dump the dataset to pickle format
    path_dump = CONFIG_DATA['data_path']
    utils.dump_pickle(data, path_dump)

    return data

def input_output_splitting():
    "Function to split the input and output as predictors and target variable"

    # Load dataset
    data_path = CONFIG_DATA['data_path']
    data = utils.load_pickle(data_path)

    # Define target variable (y)
    target_var = CONFIG_DATA['target_variable']
    y = data[target_var]

    # Define predictors variable (X)
    X = data.drop(columns = [target_var],
                  axis = 1)
    
    # Create validation to splitting X and y
    print('Shape of y :', y.shape)
    print('Shape of X :', X.shape)

    # Dump the output
    dump_path_predictors = CONFIG_DATA['predictors_set_path']
    utils.dump_pickle(X, dump_path_predictors)

    dump_path_target = CONFIG_DATA['target_set_path']
    utils.dump_pickle(y, dump_path_target)

    return X, y
    
X, y = input_output_splitting()

    # Import library 
from sklearn.model_selection import train_test_split

def train_test_splitting():
    """Function to split train & test, after that dump the data"""
    
    # Load the X and y dataset
    X = utils.load_pickle(CONFIG_DATA['predictors_set_path'])
    y = utils.load_pickle(CONFIG_DATA['target_set_path'])

    # Spliting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify = y,
                                                        test_size = CONFIG_DATA['test_size'],
                                                        random_state = 42)
    # Validation of splitting data
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump data
    utils.dump_pickle(X_train, CONFIG_DATA['train_path'][0])
    utils.dump_pickle(y_train, CONFIG_DATA['train_path'][1])
    utils.dump_pickle(X_test, CONFIG_DATA['test_path'][0])
    utils.dump_pickle(y_test, CONFIG_DATA['test_path'][1])

    return X_train, X_test, y_train, y_test

# Execute the functions
if __name__ == '__main__':
    # Load config data
    CONFIG_DATA = utils.load_config()

    # Run all functions
    open_data()
    input_output_splitting()
    train_test_splitting()