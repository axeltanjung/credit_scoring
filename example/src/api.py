# Import library
import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scaling import predict_score
from pre_processing import transform_woe


# Class for receiving the input
class api_data(BaseModel):
    Age : int
    Income : float
    Emp_length : float
    Loan_amount : float
    Int_rate : float
    Percent_income : float
    Hist_length : float
    Home : str
    Loan_intent : str
    Loan_grade : str
    Hist_default : str 


# Load config data    
CONFIG_DATA = utils.config_load()

# Create app
app = FastAPI()

# Create an address to return something (cuma uji coba)
@app.get('/')
def home():
    return "Hello world"

# Task yang diperlukan di prediction:
# 1. user bisa input data
# 2. user dapat prediksi score
# 3. user dapat prediksi probability of good
# 4. user dapat prediksi rekomendasi keputusan kredit

# Create an address to perform the prediction
@app.post('/predict')
def get_data(data: api_data):

    # Load columns list for the input
    columns_ = CONFIG_DATA['columns_']
    
    # Ingest the data input
    input_list = [
        data.Age, data.Income, data.Emp_length, 
        data.Loan_amount, data.Int_rate, data.Percent_income, 
        data.Hist_length, data.Home, data.Loan_intent, 
        data.Loan_grade, data.Hist_default
        ]
    
    # Transform the input to a dataframe
    input_table = pd.DataFrame({'0' : input_list},
                               index = columns_).T
    
    # Predict the credit score
    y_score = predict_score(raw_data = input_table,
                          CONFIG_DATA = CONFIG_DATA)
    
    # Predict the probability of good
    best_model = utils.pickle_load(CONFIG_DATA['best_model_path'])
    input_woe = transform_woe(raw_data = input_table,
                              CONFIG_DATA = CONFIG_DATA)
    y_prob = best_model.predict_proba(input_woe)[0][0]
    y_prob = round(y_prob, 2)

    # Define the recommendation (based on the credit score)
    cutoff_score = CONFIG_DATA['cutoff_score']
    if y_score > cutoff_score:
        y_status = "APPROVE"
    else:
        y_status = "REJECT"

    # Summarize the results of prediction
    results = {
        'Score' : y_score,
        'Proba' : y_prob,
        'Recommendation' : y_status
    }

    return results


if __name__ == '__main__':
    uvicorn.run('api:app',
                host = '127.0.0.1',
                port = 8000)
