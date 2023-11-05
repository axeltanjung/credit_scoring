# Import library
import pandas as pd
import utils as utils
from sklearn.model_selection import train_test_split

# Paste all functions from notebook 1.data_preparation.ipynb

# Function to load and dump the sample
def read_data():
    """Load data and dump data"""

    # Load data
    data_path = CONFIG_DATA['raw_dataset_path']
    data = pd.read_csv(data_path)

    # Validate data shape
    print("Data shape       :", data.shape)

    # Pickle dumping (save the result)
    dump_path = CONFIG_DATA['dataset_path']
    utils.pickle_dump(data, dump_path)

    return data

# Function to split input (predictors) and output (responses)
def split_input_output():
    """Split input (predictors) and output (responses)"""
    # Load data
    dataset_path = CONFIG_DATA['dataset_path']
    data = utils.pickle_load(dataset_path)

    # Define y
    response_variable = CONFIG_DATA['response_variable']
    y = data[response_variable]

    # Define X
    X = data.drop(columns = [response_variable],
                  axis = 1)
    
    # Validate the splitting
    print('y shape :', y.shape)
    print('X shape :', X.shape)

    # Dumping
    dump_path_predictors = CONFIG_DATA['predictors_set_path']
    utils.pickle_dump(X, dump_path_predictors)

    dump_path_response = CONFIG_DATA['response_set_path']    
    utils.pickle_dump(y, dump_path_response)
    
    return X, y

# Function to split train & test, then dump the data
def split_train_test():
    """Split train & test, then dump the data"""
    # Load the X and y
    X = utils.pickle_load(CONFIG_DATA['predictors_set_path'])
    y = utils.pickle_load(CONFIG_DATA['response_set_path'])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify = y,
                                                        test_size = CONFIG_DATA['test_size'],
                                                        random_state = 42)
    # Validate splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    # Dump data
    utils.pickle_dump(X_train, CONFIG_DATA['train_path'][0])
    utils.pickle_dump(y_train, CONFIG_DATA['train_path'][1])
    utils.pickle_dump(X_test, CONFIG_DATA['test_path'][0])
    utils.pickle_dump(y_test, CONFIG_DATA['test_path'][1])

    return X_train, X_test, y_train, y_test


# Execute the functions
if __name__ == '__main__':
    # Load config data
    CONFIG_DATA = utils.config_load()

    # Run all functions
    read_data()
    split_input_output()
    split_train_test()