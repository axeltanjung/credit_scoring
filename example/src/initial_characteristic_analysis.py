# Import library
import utils as utils
import pandas as pd
import numpy as np


# Paste all functions from notebook 3.initial_characteristic_analysis.ipynb
# Function to concat X and y
def concat_data(type):
    """Concat the input & output data"""
    X = utils.pickle_load(CONFIG_DATA[f'{type}_path'][0])
    y = utils.pickle_load(CONFIG_DATA[f'{type}_path'][1])
    
    # Concatenate X and y
    data = pd.concat((X, y),
                     axis = 1)

    # Validate data
    print(f'Data shape:', data.shape)

    # Dump concatenated data
    utils.pickle_dump(data, CONFIG_DATA[f'data_{type}_path'])
   
    return data

# Function for binning a numerical predictor
def create_num_binning(data, predictor_label, num_of_bins):
    """Bin the numerical predictor"""
    # Create a new column containing the binned predictor
    data[predictor_label + "_bin"] = pd.qcut(data[predictor_label],
                                             q = num_of_bins)

    return data

# Function for binning numerical columns and missing values
def bin_data(type):
    """Bin the numerical and missing data"""
    # Load the concatenated data
    data = utils.pickle_load(CONFIG_DATA[f'data_{type}_path'])

    # Bin the numerical columns
    num_columns = CONFIG_DATA['num_columns']
    num_of_bins = CONFIG_DATA['num_of_bins']

    for column in num_columns:
        data_binned = create_num_binning(data = data,
                                         predictor_label = column,
                                         num_of_bins = num_of_bins)

    # Bin missing values
    missing_columns = CONFIG_DATA['missing_columns']

    for column in missing_columns:
        # Add category 'Missing' to replace the missing values
        data_binned[column] = (data_binned[column]
                                    .cat
                                    .add_categories('Missing'))

        # Replace missing values with category 'Missing'
        data_binned[column].fillna(value = 'Missing',
                                   inplace = True)

    # Validate
    print(f"Original data shape : ", data.shape)
    print(f"Binned data shape  : ", data_binned.shape)

    # Dump binned data
    utils.pickle_dump(data_binned, CONFIG_DATA[f'data_{type}_binned_path'])
        
    return data_binned

# Function to create crosstabs
def create_crosstab_list():
    """Generate the crosstab list (contigency table) for WoE and IV calculation. Only in training data"""
    # load the binned Train data
    data_train_binned = utils.pickle_load(CONFIG_DATA['data_train_binned_path'])

    # load the response variable (we will summarize based on the response variable)
    response_variable = CONFIG_DATA['response_variable']

    # iterate over numercial columns
    crosstab_num = []
    num_columns = CONFIG_DATA['num_columns']
    for column in num_columns:
        # Create a contingency table
        crosstab = pd.crosstab(data_train_binned[column + "_bin"],
                               data_train_binned[response_variable],
                               margins = True)

        # Append to the list
        crosstab_num.append(crosstab)

    # iterate over categorical columns
    crosstab_cat = []
    cat_columns = CONFIG_DATA['cat_columns']
    for column in cat_columns:
        # Create a contingency table
        crosstab = pd.crosstab(data_train_binned[column],
                               data_train_binned[response_variable],
                               margins = True)

        # Append to the list
        crosstab_cat.append(crosstab)

    # Put all two in a crosstab_list
    crosstab_list = crosstab_num + crosstab_cat

    # Validate the crosstab_list
    print('number of num bin : ', [bin.shape for bin in crosstab_num])
    print('number of cat bin : ', [bin.shape for bin in crosstab_cat])

    # Dump the data
    utils.pickle_dump(crosstab_list, CONFIG_DATA['crosstab_list_path'])

    return crosstab_list

# Function to calculate WOE and IV
def WOE_and_IV():
    """Get the WoE and IV"""
    # Load the crosstab list
    crosstab_list = utils.pickle_load(CONFIG_DATA['crosstab_list_path'])

    # Create initial storage for WoE and IV
    WOE_list, IV_list = [], []
    
    # Perform the calculation for all crosstab list
    for crosstab in crosstab_list:
        # Calcualte the WoE and IV
        crosstab['p_good'] = crosstab[0]/crosstab[0]['All']                                 # Calculate % Good
        crosstab['p_bad'] = crosstab[1]/crosstab[1]['All']                                  # Calculate % Bad
        crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad'])                      # Calculate the WOE
        crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']   # Calculate the contribution value for IV
        IV = crosstab['contribution'][:-1].sum()                                            # Calculate the IV
        
        # Append to list
        WOE_list.append(crosstab)

        add_IV = {'Characteristic': crosstab.index.name, 
                  'Information Value': IV}
        IV_list.append(add_IV)


    # CREATE WOE TABLE
    # Create initial table to summarize the WOE values
    WOE_table = pd.DataFrame({'Characteristic': [],
                              'Attribute': [],
                              'WOE': []})
    for i in range(len(crosstab_list)):
        # Define crosstab and reset index
        crosstab = crosstab_list[i].reset_index()

        # Save the characteristic name
        char_name = crosstab.columns[0]

        # Only use two columns (Attribute name and its WOE value)
        # Drop the last row (average/total WOE)
        crosstab = crosstab.iloc[:-1, [0,-2]]
        crosstab.columns = ['Attribute', 'WOE']

        # Add the characteristic name in a column
        crosstab['Characteristic'] = char_name

        WOE_table = pd.concat((WOE_table, crosstab), 
                                axis = 0)

        # Reorder the column
        WOE_table.columns = ['Characteristic',
                            'Attribute',
                            'WOE']
    

    # CREATE IV TABLE
    # Create the initial table for IV
    IV_table = pd.DataFrame({'Characteristic': [],
                             'Information Value' : []})
    IV_table = pd.DataFrame(IV_list)

    # Define the predictive power of each characteristic
    strength = []

    # Assign the rule of thumb regarding IV
    for iv in IV_table['Information Value']:
        if iv < 0.02:
            strength.append('Unpredictive')
        elif iv >= 0.02 and iv < 0.1:
            strength.append('Weak')
        elif iv >= 0.1 and iv < 0.3:
            strength.append('Medium')
        else:
            strength.append('Strong')

    # Assign the strength to each characteristic
    IV_table = IV_table.assign(Strength = strength)

    # Sort the table by the IV values
    IV_table = IV_table.sort_values(by='Information Value')
    
    # Validate
    print('WOE table shape : ', WOE_table.shape)
    print('IV table shape  : ', IV_table.shape)

    # Dump data
    utils.pickle_dump(WOE_table, CONFIG_DATA['WOE_table_path'])
    utils.pickle_dump(IV_table, CONFIG_DATA['IV_table_path']) 

    return WOE_table, IV_table
    

# Execute the functions
if __name__ == "__main__":
    # 1. Load config file
    CONFIG_DATA = utils.config_load()

    # 2. Concat and binning the train set
    concat_data(type='train')
    bin_data(type='train')

    # 3. Obtain the WoE and IV    
    create_crosstab_list()
    WOE_and_IV()