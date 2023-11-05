# Import library
import pandas as pd
import utils as utils
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from scaling import predict_score
from pre_process import transform_woe


# Class for taking in the information
class api_data(BaseModel):
    Age : int
    Annual_Income : float
    Num_of_Loan : int
    Num_of_Delayed_Payment : int
    Outstanding_Debt : float
    Monthly_Inhand_Salary : float
    Num_Credit_Inquiries : int
    Credit_Utilization_Ratio : float
    Total_EMI_per_month : int
    Num_Bank_Accounts : int
    Num_Credit_Card : int
    Interest_Rate : float
    Delay_from_due_date : int
    Amount_invested_monthly : float
    Monthly_Balance : float
    Changed_Credit_Limit : float
    Credit_History_Age : int
    Occupation : str 
    Type_of_Loan : str
    Credit_Mix : str
    Payment_of_Min_Amount : str
    Payment_Behaviour : str

# Load config data    
CONFIG_DATA = utils.load_config()

# Create app
app = FastAPI()

# Make an address so that you may return items. 
@app.get('/')
def home():
    return "Hello world"

# Prediction tasks include: 
# 1. Input data from the user; 
# 2. Score prediction from users
# 3. Users are able to forecast the likelihood of success
# 4. Users can anticipate suggestions for credit decisions

# Make an address in order to carry out the forecast.
@app.post('/predict')
def get_data(data: api_data):

    # Load the input's list of columns
    columns_ = CONFIG_DATA['columns_']
    
    # Consume the data input
    input_list = [
        data.Age, data.Annual_Income, data.Num_of_Loan, 
        data.Num_of_Delayed_Payment, data.Outstanding_Debt, data.Monthly_Inhand_Salary, 
        data.Num_Credit_Inquiries, data.Credit_Utilization_Ratio, data.Total_EMI_per_month, 
        data.Num_Bank_Accounts, data.Num_Credit_Card, data.Interest_Rate,
        data.Delay_from_due_date, data.Amount_invested_monthly, data.Monthly_Balance,
        data.Changed_Credit_Limit, data.Credit_History_Age, data.Occupation,
        data.Type_of_Loan, data.Credit_Mix, data.Payment_of_Min_Amount, data.Payment_Behaviour
        ]
    
    # Convert the source data into a dataframe.
    input_table = pd.DataFrame({'0' : input_list},
                               index = columns_).T
    
    # Estimate your credit score.
    y_score = predict_score(raw_data = input_table,
                          CONFIG_DATA = CONFIG_DATA)
    
    # Estimate the likelihood of success
    best_model = utils.load_pickle(CONFIG_DATA['best_model_path'])
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

    # Summarize the prediction's outcomes.
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
