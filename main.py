import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Model
import lightgbm as lgb
import shap
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer

# Configuration
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

import mlflow
import mlflow.lightgbm

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-52-27-180-229.us-west-2.compute.amazonaws.com:5000/")
mlflow.set_experiment("StoreCasting Experiment")

from modules import EDA, utils, features, feature_importance, error_analysis, optimization, forecast

# Load Data
train = pd.read_csv('input/train.csv', parse_dates=['date'])
test = pd.read_csv('input/test.csv', parse_dates=['date'])
df = pd.concat([train, test], sort=False)

#EDA.eda()

# Feature selection
df = features.feature_selection(train, df)
print(df.shape)
df.to_csv('test.csv', index=False)


# Train-validation split
X_train, X_val, Y_train, Y_val, train, val, df = utils.train_validation_split(df)

# Start MLflow run for the first model
with mlflow.start_run(run_name="First_LightGBM_Model") as run:

    # First model training
    first_model = lgb.LGBMRegressor(random_state=384).fit(X_train, Y_train,
                                                          eval_metric=lambda y_true, y_pred: [utils.lgbm_smape(y_true, y_pred)])

    # Log model parameters
    mlflow.log_param("random_state", 384)
    mlflow.log_param("model", "LGBMRegressor")

    # Log SMAPE metric for train and validation sets
    train_smape = utils.smape(Y_train, first_model.predict(X_train))
    valid_smape = utils.smape(Y_val, first_model.predict(X_val))

    mlflow.log_metric("train_smape", train_smape)
    mlflow.log_metric("valid_smape", valid_smape)

    print("First Model")
    print("TRAIN SMAPE:", train_smape)
    print("VALID SMAPE:", valid_smape)
    print("###############################################################")

    # Log feature importance and other metrics
    feature_imp_df = feature_importance.feature_importance(first_model, X_train, X_val)
    #error_analysis.error_analysis(val, X_val, Y_val, first_model)

    # Select important features
    cols = feature_imp_df[feature_imp_df.gain > 0.015].feature.tolist()

    # Log the features selected
    mlflow.log_param("selected_features", cols)

    # Save the model
    mlflow.lightgbm.log_model(first_model, artifact_path="first_model")

# Start MLflow run for the second model with selected features
with mlflow.start_run(run_name="Second_LightGBM_Model") as run:

    second_model = lgb.LGBMRegressor(random_state=384).fit(
        X_train[cols], Y_train,
        eval_metric=lambda y_true, y_pred: [utils.lgbm_smape(y_true, y_pred)])

    # Log SMAPE metric for train and validation sets
    train_smape = utils.smape(Y_train, second_model.predict(X_train[cols]))
    valid_smape = utils.smape(Y_val, second_model.predict(X_val[cols]))

    mlflow.log_metric("train_smape", train_smape)
    mlflow.log_metric("valid_smape", valid_smape)

    print("Second Model")
    print("TRAIN SMAPE:", train_smape)
    print("VALID SMAPE:", valid_smape)
    print("###############################################################")

    # Save the second model
    mlflow.lightgbm.log_model(second_model, artifact_path="second_model")

# Start MLflow run for the hyperparameter-tuned model
with mlflow.start_run(run_name="Hyperparameter_Tuned_LightGBM_Model") as run:

    # Hyperparameter tuning
    best_params_ = optimization.hyperparameter_tuning(utils.smape, X_train, Y_train, cols)

    model_tuned1 = lgb.LGBMRegressor(**best_params_, random_state=384).fit(X_train[cols], Y_train)

    # Log the best hyperparameters
    mlflow.log_params(best_params_)

    # Log SMAPE metric for train and validation sets
    train_smape = utils.smape(Y_train, model_tuned1.predict(X_train[cols]))
    valid_smape = utils.smape(Y_val, model_tuned1.predict(X_val[cols]))

    mlflow.log_metric("train_smape", train_smape)
    mlflow.log_metric("valid_smape", valid_smape)

    print("Hyperparameter Tuned Model")
    print("TRAIN SMAPE:", train_smape)
    print("VALID SMAPE:", valid_smape)
    print("###############################################################")

    # Save the tuned model
    mlflow.lightgbm.log_model(model_tuned1, artifact_path="tuned_model")

    # Determine best number of iterations
    num_iterations = optimization.num_iteration(X_train, Y_train, cols, utils.lgbm_smape, X_val, Y_val, best_params_)
    mlflow.log_param("num_iterations", num_iterations)

# Start MLflow run for the final model
with mlflow.start_run(run_name="Final_LightGBM_Model") as run:

    df.sort_values(["store", "item", "date"], inplace=True)
    train_final = df.loc[(df["date"] < "2018-01-01"), :]
    test_final = df.loc[(df["date"] >= "2018-01-01"), :]

    X_train_final = train_final[cols]
    Y_train_final = train_final.sales
    X_test_final = test_final[cols]

    bucket_name = "mlflow-store-item-prediction"
    input_s3_key = "input/test.csv"
    utils.save_data_to_s3(X_test_final,bucket_name,input_s3_key)

    final_model = lgb.LGBMRegressor(**best_params_, random_state=384, metric="custom")
    final_model.set_params(n_estimators=num_iterations)
    final_model.fit(X_train_final[cols], Y_train_final,
                    eval_metric=lambda y_true, y_pred: [utils.lgbm_smape(y_true, y_pred)])

    # Log SMAPE metric for train and validation sets
    train_smape = utils.smape(Y_train, final_model.predict(X_train[cols]))
    valid_smape = utils.smape(Y_val, final_model.predict(X_val[cols]))

    mlflow.log_metric("train_smape", train_smape)
    mlflow.log_metric("valid_smape", valid_smape)

    print("Final Model")
    print("TRAIN SMAPE:", train_smape)
    print("VALID SMAPE:", valid_smape)
    print("###############################################################")

    # Log final model and forecasting output
    mlflow.lightgbm.log_model(final_model, artifact_path="final_model")

    # Forecast and log predictions (Optional)
    forecast.forecast_stores(train, X_test_final, test_final, final_model)
