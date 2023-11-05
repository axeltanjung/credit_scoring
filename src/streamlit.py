# Import library
import streamlit as st
import requests
import pandas as pd


# Create the title
st.title("Credit Score Prediction for Customer Bank Amara")
# st.subheader("Input the applicant's data and click the "Predict" button")


# Construct the input form.
with st.form(key = "applicant_data_form"):

    # Input applicant's name
    app_name = st.text_input('Applicants name', '')

    # Input person_age
    Age = st.number_input(
        label = "1.\tAge:",
        min_value = 20,
        max_value = 55,
        help = "Value range from 20 to 55"
    )

    # Input person_annual_income
    Annual_Income = st.number_input(
        label = "2.\tTotal annual income  (USD):",
        min_value = 4000.0,
        max_value = 100000000.0,
        help = "Value range from 4000.0 to 10.000.000.0"
    )

    # Input Num_of_Delayed_Payment
    Num_of_Delayed_Payment = st.number_input(
        label = "3.\tTotal of Delayed Payment that has been occured (month):",
        min_value = 0,
        max_value = 38,
        help = "Value range from 0 to 1000"
    )

    # Input Outstanding_Debt
    Outstanding_Debt = st.number_input(
        label = "4.\tTotal of oustanding debt cummulative (USD):",
        min_value = 0.0,
        max_value = 50000.0,
        help = "Value range from 0 to 50000"
    )

    # Input Monthly_Inhand_Salary
    Monthly_Inhand_Salary = st.number_input(
        label = "5.\tTotal amount of mountly inhand salary (USD):",
        min_value = 300.00,
        max_value = 20000.00,
        help = "Value range from 300.00 to 20000.00"
    )

    # Input Num_Credit_Inquiries
    Num_Credit_Inquiries = st.number_input(
        label = "6.\tTotal number of credit inquiries:",
        min_value = 0,
        max_value = 12,
        help = "Value range from 0 to 12"
    )

    # Input Credit_Utilization_Ratio
    Credit_Utilization_Ratio = st.number_input(
        label = "7.\tRatio of utilizing credit to salary (%):",
        min_value = 0,
        max_value = 30,
        help = "Value range from 20 to 60%" 
    )

    # Input Total_EMI_per_month
    Total_EMI_per_month = st.number_input(
        label = "7.\tTotal EMI per month (USD):",
        min_value = 4.0,
        max_value = 100000.0,
        help = "Value range from 4 to 100000 USD" 
    )

    # Input Num_Bank_Accounts
    Num_Bank_Accounts = st.number_input(
        label = "7.\tRepresents the number of bank accounts a person holds",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10" 
    )

    # Input Num_Credit_Card
    Num_Credit_Card = st.number_input(
        label = "7.\tRepresents the number of other credit cards held by a person",
        min_value = 0,
        max_value = 10,
        help = "Value range from 0 to 10" 
    )

    # Input Interest_Rate
    Interest_Rate = st.number_input(
        label = "7.\tRepresents the interest rate on credit card (%)",
        min_value = 1,
        max_value = 40,
        help = "Value range from 1 to 40" 
    )

    # Input Delay_from_due_date
    Delay_from_due_date = st.number_input(
        label = "7.\tRepresents the average number of days delayed from the payment date",
        min_value = 0,
        max_value = 70,
        help = "Value range from 0 to 70 days" 
    )

    # Input Amount_invested_monthly
    Amount_invested_monthly = st.number_input(
        label = "7.\tRepresents the monthly amount invested by the customer (in USD)",
        min_value = 0.0,
        max_value = 10000.0,
        help = "Value range from 0 to 10000.0 USD" 
    )
    
    # Input Monthly_Balance
    Monthly_Balance = st.number_input(
        label = "7.\tRepresents the monthly balance amount of the customer (in USD)",
        min_value = 0.0,
        max_value = 20000.0,
        help = "Value range from 0 to 20000.0 USD" 
    )

    # Input Changed_Credit_Limit
    Changed_Credit_Limit = st.number_input(
        label = "7.\tRepresents the percentage change in credit card limit (%)",
        min_value = -10.0,
        max_value = 40.0,
        help = "Value range from -10 to 40%" 
    )

    # Input Credit_History_Age
    Credit_History_Age = st.number_input(
        label = "7.\Represents the age of credit history of the person",
        min_value = 0.0,
        max_value = 50.0,
        help = "Value range from 0 to 50 month" 
    )

    # Input Occupation
    Occupation = st.selectbox(
        label = "9. \tRepresents the occupation of the person:",
        options = ("Scientist", "Entrepreneur", "Lawyer", "Lawyer", 
                   "Teacher", "Engineer", "Writer", "Architect", 
                   "Musician", "Developer", "Manager", "Journalist",
                   "Media_Manager", "Mechanic", "Accountant", "_______")
    ) 

    # Input Type_of_Loan
    Type_of_Loan = st.selectbox(
        label = "9. \tRepresents the types of loan taken by a person:",
        options = ("Auto Loan", "Credit-Builder Loan", "Credit-Builder Loan", "Mortgage Loan", 
                   "Payday Loan", "Student Loan", "Home Equity Loan", "Debt Consolidation Loan", 
                   "Others Loan")
    ) 

    # Input Credit_Mix
    Credit_Mix = st.selectbox(
        label = "9. \tRepresents the classification of the mix of credits:",
        options = ("Good", "Standard", "Poor")
    ) 

    # Input Payment_of_Min_Amount
    Payment_of_Min_Amount = st.selectbox(
        label = "9. \tRepresents whether only the minimum amount was paid by the person:",
        options = ("Yes", "No", "NM")
    ) 

    # Input Payment_Behaviour
    Payment_Behaviour = st.selectbox(
        label = "9. \tRepresents the payment behavior of the customer (in USD)",
        options = ("Low_spent_Small_value_payments", "Low_spent_Large_value_payments", "Low_spent_Medium_value_payments",
                   "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments",
                   "High_spent_Large_value_payments")
    )

    # Create the submit button
    submitted = st.form_submit_button("PREDICT")

    # Condition if the input is submitted
    if submitted:
        # Collect the data
        applicant_data_form = {
            "Age": Age,
            "Annual_Income": Annual_Income,
            "Num_of_Delayed_Payment": Num_of_Delayed_Payment,
            "Outstanding_Debt": Outstanding_Debt,
            "Monthly_Inhand_Salary": Monthly_Inhand_Salary,
            "Num_Credit_Inquiries": Num_Credit_Inquiries,
            "Credit_Utilization_Ratio": Credit_Utilization_Ratio,
            "Total_EMI_per_month": Total_EMI_per_month,
            "Num_Bank_Accounts": Num_Bank_Accounts,
            "Num_Credit_Card": Num_Credit_Card,
            "Interest_Rate": Interest_Rate,
            "Delay_from_due_date": Delay_from_due_date,
            "Amount_invested_monthly": Amount_invested_monthly,
            "Monthly_Balance": Monthly_Balance,
            "Changed_Credit_Limit": Changed_Credit_Limit,
            "Credit_History_Age": Credit_History_Age,
            "Occupation": Occupation,
            "Type_of_Loan": Type_of_Loan,
            "Credit_Mix": Credit_Mix,
            "Payment_of_Min_Amount": Payment_of_Min_Amount,
            "Payment_Behaviour": Payment_Behaviour
        }

        # Animate the loading process to transmit the data.
        with st.spinner("Sent data to server and predict the customer..."):
            res = requests.post("http://localhost:8000/predict",
                                json = applicant_data_form).json()
        # Print the results
        st.write(res)

        st.success(f"""
                Applicant's name for credit card: **{app_name}**
                     
                Estimation of Credit score: **{res['Score']}**  
                Probability of being good: **{res['Proba']}**  
                Recommendation: **{res['Recommendation']}**
            """)
