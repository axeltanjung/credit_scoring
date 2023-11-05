# Import library
import pandas as pd
import numpy as np

# Load the data configuration
import utils as utils

CONFIG_DATA = utils.load_config()

def data_concat(type):
    """Function to concat the input (X) & output (y) data"""
    X = utils.load_pickle(CONFIG_DATA[f'{type}_path'][0])
    y = utils.load_pickle(CONFIG_DATA[f'{type}_path'][1])
    
    # Concatenate data X and y
    data = pd.concat((X, y),
                     axis = 1)

    # Validation dataset
    print(f'Shape of data:', data.shape)

    # Dump concatenated data
    utils.dump_pickle(data, CONFIG_DATA[f'data_{type}_path'])
   
    return data

# Make a function that allows the numerical predictor to be binned.
def define_num_binning(data, predictor_label, num_of_bins):
    """Create function to binning the numerical predictor"""
    # Make a new column with the binned predictor in it.
    data[predictor_label + "_bin"] = pd.qcut(data[predictor_label],
                                             q = num_of_bins)

    return data

def data_bin(type):
    """Create function to binning the numerical and missing data"""
    # Load the concatenated data
    data = utils.load_pickle(CONFIG_DATA[f'data_{type}_path'])

    # Bin the numerical columns
    num_columns = CONFIG_DATA['num_columns']
    num_of_bins = CONFIG_DATA['num_of_bins']

    for column in num_columns:
        data_binned = define_num_binning(data = data,
                                         predictor_label = column,
                                         num_of_bins = num_of_bins)

    # Bin missing values
    missing_columns = CONFIG_DATA['missing_columns']

    for column in missing_columns:
        # Incorporate the 'Missing' category to substitute the absent values.
        data_binned[column] = (data_binned[column]
                                    .cat
                                    .add_categories('Missing'))

        # Replace missing values with category 'Missing'
        data_binned[column].fillna(value = 'Missing',
                                   inplace = True)

    # Validate
    print(f"The original data shape : ", data.shape)
    print(f"The binned data shape  : ", data_binned.shape)

    # Dump binned data
    utils.dump_pickle(data_binned, CONFIG_DATA[f'data_{type}_binned_path'])
        
    return data_binned

def define_crosstab_list():
    """Create the crosstab list (contingency table) needed for the computation of WOE and IV. In training data only"""
    # load the binned train data
    data_train_binned = utils.load_pickle(CONFIG_DATA['data_train_binned_path'])

    # the target variable is loaded (The target variable will be used to summarize.)
    target_variable = CONFIG_DATA['target_variable']

    # Repeat over the numerical columns.
    crosstab_num = []
    num_columns = CONFIG_DATA['num_columns']
    for column in num_columns:
        # Establish a contigency table.
        crosstab = pd.crosstab(data_train_binned[column + "_bin"],
                               data_train_binned[target_variable],
                               margins = True)

        # Append to the list
        crosstab_num.append(crosstab)

    # Repeat with the category columns.
    crosstab_cat = []
    cat_columns = CONFIG_DATA['cat_columns']
    for column in cat_columns:
        # Establish a contigency table.
        crosstab = pd.crosstab(data_train_binned[column],
                               data_train_binned[target_variable],
                               margins = True)

        # Append to the list
        crosstab_cat.append(crosstab)

    # Put all two in a crosstab_list
    crosstab_list = crosstab_num + crosstab_cat

    # Validate the crosstab_list
    print('Count of num bin : ', [bin.shape for bin in crosstab_num])
    print('Count of cat bin : ', [bin.shape for bin in crosstab_cat])

    # Dump the result
    utils.dump_pickle(crosstab_list, CONFIG_DATA['crosstab_list_path'])

    return crosstab_list

def WOE_and_IV():
    """Obtain the IV and WoE."""
    # Load the crosstab list
    crosstab_list = utils.load_pickle(CONFIG_DATA['crosstab_list_path'])

    # Establish initial IV and WoE storage
    WOE_list, IV_list = [], []
    
    # Execute the computation for every crosstab list.
    for crosstab in crosstab_list:
        # Determine the WoE and IV.
        crosstab['p_good'] = crosstab[1]/crosstab[1]['All']                                 # Calculate % Good
        crosstab['p_bad'] = crosstab[0]/crosstab[0]['All']                                  # Calculate % Bad
        crosstab['WOE'] = np.log(crosstab['p_good']/crosstab['p_bad'])                      # Calculate the WOE
        crosstab['contribution'] = (crosstab['p_good']-crosstab['p_bad'])*crosstab['WOE']   # Calculate the contribution value for IV
        IV = crosstab['contribution'][:-1].sum()                                            # Calculate the IV
        
        # Append to list
        WOE_list.append(crosstab)

        add_IV = {'Characteristic': crosstab.index.name, 
                  'Information Value': IV}
        IV_list.append(add_IV)


    # CREATE WOE TABLE
    # Make the first table outlining the WOE values.
    WOE_table = pd.DataFrame({'Characteristic': [],
                              'Attribute': [],
                              'WOE': []})
    for i in range(len(crosstab_list)):
        # Crosstab definition and index reset
        crosstab = crosstab_list[i].reset_index()

        # Keep the trait name saved.
        char_name = crosstab.columns[0]

        # Utilize just the attribute name and its WOE value in two columns.
        # Remove the final row (average/total WOE).
        crosstab = crosstab.iloc[:-1, [0,-2]]
        crosstab.columns = ['Attribute', 'WOE']

        # Put the name of the characteristic in a column.
        crosstab['Characteristic'] = char_name

        WOE_table = pd.concat((WOE_table, crosstab), 
                                axis = 0)

        # Reposition the column.
        WOE_table.columns = ['Characteristic',
                            'Attribute',
                            'WOE']
    

    # CREATE IV TABLE
    # Make the IV initial table.
    IV_table = pd.DataFrame({'Characteristic': [],
                             'Information Value' : []})
    IV_table = pd.DataFrame(IV_list)

    # Describe each characteristic's capacity for prediction.
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

    # Assign each characteristic a strength.
    IV_table = IV_table.assign(Strength = strength)

    # Table sorted according to IV values
    IV_table = IV_table.sort_values(by='Information Value')
    
    # Validation
    print('WOE table shape : ', WOE_table.shape)
    print('IV table shape  : ', IV_table.shape)

    # Dump data
    utils.dump_pickle(WOE_table, CONFIG_DATA['WOE_table_path'])
    utils.dump_pickle(IV_table, CONFIG_DATA['IV_table_path']) 

    return WOE_table, IV_table

WOE_table, IV_table = WOE_and_IV()

# Execute the functions
if __name__ == "__main__":
    # 1. Load config file
    CONFIG_DATA = utils.load_config()

    # 2. Concat and binning the train set
    data_concat(type='train')
    data_bin(type='train')

    # 3. Obtain the WoE and IV    
    define_crosstab_list()
    WOE_and_IV()