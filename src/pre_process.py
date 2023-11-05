# Import library
import pandas as pd
import numpy as np

# Load configuration
import utils as utils

CONFIG_DATA = utils.load_config()

# Mechanism for producing the WOE mapping dictionary
def get_woe_map_dict():
    """Obtain the WOE mapping directory."""
    # Load the WOE table
    WOE_table = utils.load_pickle(CONFIG_DATA['WOE_table_path'])

    # Set the dictionary to start.
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}
    
    unique_char = set(WOE_table['Characteristic'])
    
    for char in unique_char:
        # Obtain the WOE and attribute information for each characteristic.
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     # Utilize a characteristic-based filter
                            [['Attribute', 'WOE']])                 # Next, choose WOE and the attribute.
        
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

    # Validation of data
    print('Number of key : ', len(WOE_map_dict.keys()))

    # Dump
    utils.dump_pickle(WOE_map_dict, CONFIG_DATA['WOE_map_dict_path'])

    return WOE_map_dict

# function to insert WOE values in place of the train set's raw data
def transform_woe(raw_data=None, type=None, CONFIG_DATA=None):
    """Substitute WOE for the data value."""
    # Load the numerical columns
    num_cols = CONFIG_DATA['num_columns']

    # Load the WOE_map_dict
    WOE_map_dict = utils.load_pickle(CONFIG_DATA['WOE_map_dict_path'])

    # In case type is not None, load the stored data.
    if type is not None:
        raw_data = utils.load_pickle(CONFIG_DATA[f'{type}_path'][0])

    # Map the data
    woe_data = raw_data.copy()
    for col in woe_data.columns:
        # Fix numerical columns
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col    

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    # Check the data to see if any values are missing or outside of the range.
    for col in woe_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col 

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])
        woe_data['Monthly_Balance']=woe_data['Monthly_Balance'].fillna(value=0)

    # Validate
    print('Raw data shape : ', raw_data.shape)
    print('WOE data shape : ', woe_data.shape)

    # Dump data
    if type is not None:
        utils.dump_pickle(woe_data, CONFIG_DATA[f'X_{type}_woe_path'])

    return woe_data

# Execute the functions
if __name__ == "__main__":
    # 1. Load config file
    CONFIG_DATA = utils.load_config()

    # 2. Generate the WOE map dict
    get_woe_map_dict()

    # 3. Transform the raw train set into WOE values
    transform_woe(type='train', CONFIG_DATA=CONFIG_DATA)