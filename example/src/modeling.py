# Import library
import utils as utils
import pandas as pd
import numpy as np

# Import library for modeling
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

# Paste all functions from notebook 5. modeling.ipynb

# Function to perform forward selection procedure
def forward(X, y, predictors, scoring='roc_auc', cv=5):
    """Function to perform forward selection procedure"""

    # Define sample size and  number of all predictors
    n_samples, n_predictors = X.shape

    # Define list of all predictors
    col_list = np.arange(n_predictors)

    # Define remaining predictors for each k
    remaining_predictors = [p for p in col_list if p not in predictors]

    # Initialize list of predictors and its CV Score
    pred_list = []
    score_list = []

    # Cross validate each possible combination of remaining predictors
    for p in remaining_predictors:
        combi = predictors + [p]

        # Extract predictors combination
        X_ = X[:, combi]
        y_ = y

        # Define the estimator
        model = LogisticRegression(penalty = None,
                                   class_weight = 'balanced')

        # Cross validate the recall scores of the model
        cv_results = cross_validate(estimator = model,
                                    X = X_,
                                    y = y_,
                                    scoring = scoring,
                                    cv = cv)

        # Calculate the average CV/recall score
        score_ = np.mean(cv_results['test_score'])

        # Append predictors combination and its CV Score to the list
        pred_list.append(list(combi))
        score_list.append(score_)

    # Tabulate the results
    models = pd.DataFrame({"Predictors": pred_list,
                           "CV Score": score_list})

    # Choose the best model
    best_model = models.loc[models['CV Score'].argmax()]

    return models, best_model

# Function to perform forward selection on all characteristics
def run_forward():
    """Function to perform forward selection on all characteristics"""

    cv = CONFIG_DATA['num_of_cv']
    scoring = CONFIG_DATA['scoring']

    X_train_woe_path = CONFIG_DATA['X_train_woe_path']
    X_train_woe = utils.pickle_load(X_train_woe_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = CONFIG_DATA['train_path'][1]
    y_train = utils.pickle_load(y_train_path)
    y_train = y_train.to_numpy()

    # First, fit the null model
    # Define predictor for the null model
    predictor = []

    # The predictor in the null model is zero values for all predictors
    X_null = np.zeros((X_train.shape[0], 1))

    # Define the estimator
    model = LogisticRegression(penalty = None,
                               class_weight = 'balanced')

    # Cross validate
    cv_results = cross_validate(estimator = model,
                                X = X_null,
                                y = y_train,
                                cv = cv,
                                scoring = scoring)

    # Calculate the average CV score
    score_ = np.mean(cv_results['test_score'])

    # Create table for the best model of each k predictors
    # Append the results of null model
    forward_models = pd.DataFrame({"Predictors": [predictor],
                                   "CV Score": [score_]})

    # Next, perform forward selection for all predictors
    # Define list of predictors
    predictors = []
    n_predictors = X_train.shape[1]

    # Perform forward selection procedure for k=1,...,n_predictors
    for k in range(n_predictors):
        _, best_model = forward(X = X_train,
                                y = y_train,
                                predictors = predictors,
                                scoring = scoring,
                                cv = cv)

        # Tabulate the best model of each k predictors
        forward_models.loc[k+1] = best_model
        predictors = best_model['Predictors']

    # Find the best CV score
    best_idx = forward_models['CV Score'].argmax()
    best_cv_score = forward_models['CV Score'].loc[best_idx]
    best_predictors = forward_models['Predictors'].loc[best_idx]

    # Print the summary
    print('===================================================')
    print('Best index            :', best_idx)
    print('Best CV Score         :', best_cv_score)
    print('Best predictors (idx) :', best_predictors)
    print('Best predictors       :')
    print(X_train_woe.columns[best_predictors].tolist())
    print('===================================================')

    print(forward_models)
    print('===================================================')
    
    forward_models_path = CONFIG_DATA['forward_models_path']
    utils.pickle_dump(forward_models, forward_models_path)

    best_predictors_path = CONFIG_DATA['best_predictors_path']
    utils.pickle_dump(best_predictors, best_predictors_path)

    return forward_models, best_predictors

# Function to fit best model on whole X_train
def best_model_fitting(best_predictors):
    """Function to fit best model on whole X_train"""

    X_train_path = CONFIG_DATA['X_train_woe_path']
    X_train_woe = utils.pickle_load(X_train_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = CONFIG_DATA['train_path'][1]
    y_train = utils.pickle_load(y_train_path)
    y_train = y_train.to_numpy()

    if best_predictors is None:
        best_predictors_path = CONFIG_DATA['best_predictors_path']
        best_predictors = utils.pickle_load(best_predictors_path)
        print(f"Best predictors index   :", best_predictors)
    else:
        print(f"[Adjusted] best predictors index   :", best_predictors)

    # Define X with best predictors
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty = None,
                                    class_weight = 'balanced')
    best_model.fit(X_train_best, y_train)

    print(best_model)

    # Extract the best model' parameter estimates
    best_model_intercept = pd.DataFrame({'Characteristic': 'Intercept',
                                         'Estimate': best_model.intercept_})
    
    best_model_params = X_train_woe.columns[best_predictors].tolist()

    best_model_coefs = pd.DataFrame({'Characteristic': best_model_params,
                                     'Estimate': np.reshape(best_model.coef_, 
                                                            len(best_predictors))})

    best_model_summary = pd.concat((best_model_intercept, best_model_coefs),
                                   axis = 0,
                                   ignore_index = True)
    
    print('===================================================')
    print(best_model_summary)
    
    best_model_path = CONFIG_DATA['best_model_path']
    utils.pickle_dump(best_model, best_model_path)

    best_model_summary_path = CONFIG_DATA['best_model_summary_path']
    utils.pickle_dump(best_model_summary, best_model_summary_path)

    return best_model, best_model_summary

# Execute the functions
if __name__ == "__main__":

    # 1. Load config file
    CONFIG_DATA = utils.config_load()

    # 2. Perform model selection on all characteristics
    run_forward()

    # 3. Fit the best model (with adjustment)
    best_model_fitting(best_predictors = [0,1,2,3,4,5,6,7,8,9,10])