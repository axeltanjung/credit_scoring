# Import library
import pandas as pd
import numpy as np

# Import library for modeling
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

# Load configuration
import utils as utils

CONFIG_DATA = utils.load_config()

# The forward selection procedure's function
def forward(X, y, predictors, scoring='roc_auc', cv=5):
    """Function for carrying out the forward selection process"""

    # Specify the number of all predictors and the sample size.
    n_samples, n_predictors = X.shape

    # Describe the entire predictor list.
    col_list = np.arange(n_predictors)

    # For every k, define the remaining predictors.
    remaining_predictors = [p for p in col_list if p not in predictors]

    # Set the CV Score and predictors' initial values.
    pred_list = []
    score_list = []

    # Every possible pairing of the remaining predictors should be cross-validated.
    for p in remaining_predictors:
        combi = predictors + [p]

        # Combine extract predictors
        X_ = X[:, combi]
        y_ = y

        # Define the estimator
        model = LogisticRegression(penalty = 'l2',
                                   class_weight = 'balanced')

        # Cross-validate the model's recall scores
        cv_results = cross_validate(estimator = model,
                                    X = X_,
                                    y = y_,
                                    scoring = scoring,
                                    cv = cv)

        # Determine the typical CV/recall score.
        score_ = np.mean(cv_results['test_score'])

        # Add the combination of predictors and their CV score to the list.
        pred_list.append(list(combi))
        score_list.append(score_)

    # Total the outcomes.
    models = pd.DataFrame({"Predictors": pred_list,
                           "CV Score": score_list})

    # Select the best model.
    best_model = models.loc[models['CV Score'].argmax()]

    return models, best_model

# The ability to carry out forward selection across all attributes
def run_forward():
    """Function to carry out forward selection based on every attribute"""

    cv = CONFIG_DATA['num_of_cv']
    scoring = CONFIG_DATA['scoring']

    X_train_woe_path = CONFIG_DATA['X_train_woe_path']
    X_train_woe = utils.load_pickle(X_train_woe_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = CONFIG_DATA['train_path'][1]
    y_train = utils.load_pickle(y_train_path)
    y_train = y_train.to_numpy()

    # First, fit the null model
    # Define the null model's predictor.
    predictor = []

    # In the null model, every predictor has a value of zero.
    X_null = np.zeros((X_train.shape[0], 1))

    # Define the estimator
    model = LogisticRegression(penalty = 'l2',
                               class_weight = 'balanced')

    # Cross validate
    cv_results = cross_validate(estimator = model,
                                X = X_null,
                                y = y_train,
                                cv = cv,
                                scoring = scoring)

    # Determine the typical CV score.
    score_ = np.mean(cv_results['test_score'])

    # Make a table with each k predictor's best model.
    # Add the null model results.
    forward_models = pd.DataFrame({"Predictors": [predictor],
                                   "CV Score": [score_]})

    # Proceed with forward selection for each and every predictor.
    # Define the predictor list.
    predictors = []
    n_predictors = X_train.shape[1]

    # Apply the forward selection method to the predictors k=1,...,n_predictors.
    for k in range(n_predictors):
        _, best_model = forward(X = X_train,
                                y = y_train,
                                predictors = predictors,
                                scoring = scoring,
                                cv = cv)

        # List the optimal model for each of the k predictors.
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
    utils.dump_pickle(forward_models, forward_models_path)

    best_predictors_path = CONFIG_DATA['best_predictors_path']
    utils.dump_pickle(best_predictors, best_predictors_path)

    return forward_models, best_predictors

# Function to fit optimal model across the entire X_train
def best_model_fitting(best_predictors):
    """Function to fit optimal model across the entire X_train"""

    X_train_path = CONFIG_DATA['X_train_woe_path']
    X_train_woe = utils.load_pickle(X_train_path)
    X_train = X_train_woe.to_numpy()

    y_train_path = CONFIG_DATA['train_path'][1]
    y_train = utils.load_pickle(y_train_path)
    y_train = y_train.to_numpy()

    if best_predictors is None:
        best_predictors_path = CONFIG_DATA['best_predictors_path']
        best_predictors = utils.load_pickle(best_predictors_path)
        print(f"Best predictors index   :", best_predictors)
    else:
        print(f"[Adjusted] best predictors index   :", best_predictors)

    # Use the best predictors to define X.
    X_train_best = X_train[:, best_predictors]

    # Fit best model
    best_model = LogisticRegression(penalty = 'l2',
                                    class_weight = 'balanced')
    best_model.fit(X_train_best, y_train)

    print(best_model)

    # Extract parameter estimates from the optimal model.
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
    utils.dump_pickle(best_model, best_model_path)

    best_model_summary_path = CONFIG_DATA['best_model_summary_path']
    utils.dump_pickle(best_model_summary, best_model_summary_path)

    return best_model, best_model_summary

# Execute the functions
if __name__ == "__main__":

    # 1. Load config file
    CONFIG_DATA = utils.load_config()

    # 2. Perform model selection on all characteristics
    run_forward()

    # 3. Fit the best model (with adjustment)
    best_model_fitting(best_predictors = [0,1,2,3,4,5,6,7,8,9,10,11])
