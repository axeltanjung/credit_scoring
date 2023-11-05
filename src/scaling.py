# Import library
import pandas as pd
import numpy as np

# Load configuration
import utils as utils

# Update the config file 
CONFIG_DATA = utils.load_config()

# The function that transforms the model's output into points
def scaling():
    """The function that transforms the model's output into points"""

    # Describe the references (pdo, score, and odds).
    pdo = CONFIG_DATA['pdo']
    score = CONFIG_DATA['score_ref']
    odds = CONFIG_DATA['odds_ref']

    # Load the best model
    best_model_path = CONFIG_DATA['best_model_path']
    best_model = utils.load_pickle(best_model_path)

    # Load the WOE table
    WOE_table_path = CONFIG_DATA['WOE_table_path']
    WOE_table = utils.load_pickle(WOE_table_path)

    # Load the best model's estimates table
    best_model_summary_path = CONFIG_DATA['best_model_summary_path']
    best_model_summary = utils.load_pickle(best_model_summary_path)

    # Calculate Factor and Offset
    factor = pdo/np.log(2)
    offset = score-(factor*np.log(odds))

    print('===================================================')
    print(f"Odds of good of {odds}:1 at {score} points score.")
    print(f"{pdo} PDO (points to double the odds of good).")
    print(f"Offset = {offset:.2f}")
    print(f"Factor = {factor:.2f}")
    print('===================================================')

    # Define n = number of characteristics
    n = best_model_summary.shape[0] - 1

    # Define b0
    b0 = best_model.intercept_[0]

    # Change the name of the characteristic in best_model_summary_table.
    num_cols = CONFIG_DATA['num_columns']
    for col in best_model_summary['Characteristic']:

        if col in num_cols:
            bin_col = col + '_bin'
        else:
            bin_col = col

        best_model_summary.replace(col, bin_col, inplace = True) 

    # To obtain a beta or parameter estimate for each characteristic, merge tables.
    scorecards = pd.merge(left = WOE_table,
                          right = best_model_summary,
                          how = 'left',
                          on = ['Characteristic'])
    
    # Define beta and WOE
    beta = scorecards['Estimate']
    WOE = scorecards['WOE']

    # Determine the point value for every attribute.
    scorecards['Points'] = round((offset/n) - factor*((b0/n) + (beta*WOE)))
    try :
        scorecards['Points'] = scorecards['Points'].astype('int')
    except Exception:
        pass

    # Validation
    print('Scorecards table shape : ', scorecards.shape)
    
    # Dump the scorecards
    scorecards_path = CONFIG_DATA['scorecards_path']
    utils.dump_pickle(scorecards, scorecards_path)

    return scorecards

# Create the dict function for the Points map.
def get_points_map_dict():
    """Create the dict function for the Points map."""
    # Load the Scorecards table
    scorecards = utils.load_pickle(CONFIG_DATA['scorecards_path'])

    # Set the dictionary to start.
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        # Obtain the WOE and attribute information for each characteristic.
        current_data = (scorecards
                            [scorecards['Characteristic']==char]     # Filter based on characteristic
                            [['Attribute', 'Points']])                 # Then select the attribute & WOE
        
        # Get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']

            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan

    # Validation data
    print('Number of key : ', len(points_map_dict.keys()))

    # Dump
    utils.dump_pickle(points_map_dict, CONFIG_DATA['points_map_dict_path'])

    return points_map_dict
    
def transform_points(raw_data=None, type=None, CONFIG_DATA=None):
    """Swap out the data value for points."""
    # Load the numerical columns
    num_cols = CONFIG_DATA['num_columns']

    # Load the points_map_dict
    points_map_dict = utils.load_pickle(CONFIG_DATA['points_map_dict_path'])

    # Load the saved data if type is not None
    if type is not None:
        raw_data = utils.load_pickle(CONFIG_DATA[f'{type}_path'][0])

    # Map the data
    points_data = raw_data.copy()
    for col in points_data.columns:
        # Fix numerical columns
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col    

        points_data[col] = points_data[col].map(points_map_dict[map_col])

    # If a value is missing or outside of the range, map the data.
    for col in points_data.columns:
        if col in num_cols:
            map_col = col + '_bin'
        else:
            map_col = col 

        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])

    # Dump data
    if type is not None:
        utils.dump_pickle(points_data, CONFIG_DATA[f'X_{type}_points_path'])

    return points_data

# Predictive function for credit score
def predict_score(raw_data, CONFIG_DATA):
    """Predictive function for credit score"""
    
    points = transform_points(raw_data = raw_data, 
                              type = None, 
                              CONFIG_DATA = CONFIG_DATA)
    
    score = int(points.sum(axis=1))

    # print(f"Credit Score : ", score)
    
    # cutoff_score = CONFIG_DATA['cutoff_score']

    # if score > cutoff_score:
    #     print("Recommendation : APPROVE")
    # else:
    #     print("Recommendation : REJECT")

    utils.dump_pickle(score, CONFIG_DATA['score_path'])

    return score

# Execute the functions
if __name__ == "__main__":

    # 1. Load config file
    CONFIG_DATA = utils.load_config()

    # 2. Create the scorecards
    scaling()

    # 3. Generate the points map dict
    get_points_map_dict()

    # 4. Predict the score
    transform_points(type='train', CONFIG_DATA=CONFIG_DATA)
