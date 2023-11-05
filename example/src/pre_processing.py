# Import library
import pandas as pd
import numpy as np
import utils as utils

# Paste all functions from notebook 4. pre_processing.ipynb

# Function to generate the WOE mapping dictionary
def get_woe_map_dict():
    """Get the WOE mapping dictionary"""
    # Load the WOE table
    WOE_table = utils.pickle_load(CONFIG_DATA['WOE_table_path'])

    # Initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}
    
    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        # Get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'WOE']])                 # Then select the attribute & WOE
        
        # Get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    # Validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    # Dump
    utils.pickle_dump(WOE_map_dict, CONFIG_DATA['WOE_map_dict_path'])

    return WOE_map_dict

# Function to replace the raw data in the train set with WOE values
def transform_woe(raw_data=None, type=None, CONFIG_DATA=None):
    """Replace data value with WOE"""
    # Load the numerical columns
    num_cols = CONFIG_DATA['num_columns']

    # Load the WOE_map_dict
    WOE_map_dict = utils.pickle_load(CONFIG_DATA['WOE_map_dict_path'])

    # Load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(CONFIG_DATA[f'{type}_path'][0])

    # Map the data
    woe_data = raw_data.copy()
    for col in woe_data.columns:
        # Perbaiki kolom numerik
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col    

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Map the data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col 

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    # Validate
    print('Raw data shape : ', raw_data.shape)
    print('WOE data shape : ', woe_data.shape)

    # Dump data
    if type is not None:
        utils.pickle_dump(woe_data, CONFIG_DATA[f'X_{type}_woe_path'])

    return woe_data


# Execute the functions
if __name__ == "__main__":
    # 1. Load config file
    CONFIG_DATA = utils.config_load()

    # 2. Generate the WOE map dict
    get_woe_map_dict()

    # 3. Transform the raw train set into WOE values
    transform_woe(type='train', CONFIG_DATA=CONFIG_DATA)