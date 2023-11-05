# Import library
import streamlit as st
import requests
import pandas as pd


# Create the title
st.title("Credit Score Prediction")
# st.subheader("Input the applicant's data and click the "Predict" button")


# Create the form for input
with st.form(key = "applicant_data_form"):

    # Input applicant's name
    app_name = st.text_input('Applicants name', '')

    # Input person_age
    person_age = st.number_input(
        label = "1.\tAge:",
        min_value = 20,
        max_value = 55,
        help = "Value range from 20 to 55"
    )

    # Input person_income
    person_income = st.number_input(
        label = "2.\tTotal annual income  (USD):",
        min_value = 4000.0,
        max_value = 6000000.0,
        help = "Value range from 4000.0 to 6000000.0"
    )

    # Input person_emp_length
    person_emp_length = st.number_input(
        label = "3.\tEmployment length (year):",
        min_value = 0,
        max_value = 38,
        help = "Value range from 0 to 38"
    )

    # Input loan_amnt
    loan_amnt = st.number_input(
        label = "4.\tLoan amount (USD):",
        min_value = 500.0,
        max_value = 35000.0,
        help = "Value range from 500 to 35000"
    )

    # Input loan_int_rate
    loan_int_rate = st.number_input(
        label = "5.\tLoan interest rate (%):",
        min_value = 5.00,
        max_value = 24.00,
        help = "Value range from 5.00 to 24.00"
    )

    # Input loan_percent_income
    loan_percent_income = st.number_input(
        label = "6.\tIncome ratio:",
        min_value = 0.0,
        max_value = 1.0,
        help = "Value range from 0 to 1"
    )

    # Input cb_person_cred_hist_length
    cb_person_cred_hist_length = st.number_input(
        label = "7.\tCredit history length (year):",
        min_value = 0,
        max_value = 30,
        help = "Value range from 0 to 30 tahun" 
    )

    # Input person_home_ownership
    person_home_ownership = st.radio(
        label = "8. \tResidential status:",
        options = ["RENT", "MORTGAGE", "OWN", "OTHER"],
        index = 0,
        horizontal = True
    )

    # Input loan_intent
    loan_intent = st.selectbox(
        label = "9. \tLoan purpose:",
        options = ("EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", 
                   "DEBT CONSOLIDATION", "HOME IMPROVEMENT")
    ) 

    # Input loan_grade
    loan_grade = st.radio(
        label = "9. \tLoan grade:",
        options = ["A", "B", "C", "D", "E", "F", "G"],
        index = 0,
        horizontal = True
    ) 

    # Input cb_person_default_on_file
    cb_person_default_on_file = st.radio(
        label = "9. \tHas a history of default?",
        options = ["N", "Y"],
        captions = ["No", "Yes"],
        index = 0,
        horizontal = True
    )

    # Create the submit button
    submitted = st.form_submit_button("PREDICT")

    # Condition if the input is submitted
    if submitted:
        # Collect the data
        applicant_data_form = {
            "Age": person_age,
            "Income": person_income,
            "Emp_length": person_emp_length,
            "Loan_amount": loan_amnt,
            "Int_rate": loan_int_rate,
            "Percent_income": loan_percent_income,
            "Hist_length": cb_person_cred_hist_length,
            "Home": person_home_ownership,
            "Loan_intent": loan_intent,
            "Loan_grade": loan_grade,
            "Hist_default": cb_person_default_on_file
        }

        # Create a loading animation to send the data
        with st.spinner("Kirim data untuk diprediksi server ..."):
            res = requests.post("http://localhost:8000/predict",
                                json = applicant_data_form).json()
        # Print the results
        st.write(res)

        st.success(f"""
                Applicant's name: **{app_name}**
                     
                Credit score: **{res['Score']}**  
                Probability of being good: **{res['Proba']}**  
                Recommendation: **{res['Recommendation']}**
            """)
