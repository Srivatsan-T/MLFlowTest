import mlflow
import mlflow.pyfunc
import boto3
import pandas as pd
import os
from io import StringIO

mlflow.set_tracking_uri("http://ec2-52-27-180-229.us-west-2.compute.amazonaws.com:5000")  

# Configuration
s3_client = boto3.client('s3')
bucket_name = "mlflow-store-item-prediction"
input_s3_key = "input/test.csv"  # S3 path for input data
output_s3_key = "output/predictions.csv"  # S3 path to store predictions


def load_data_from_s3(bucket_name, s3_key):
    """Load data from an S3 bucket."""
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    if status == 200:
        print(f"Successfully fetched data from {bucket_name}/{s3_key}")
        data = pd.read_csv(response.get("Body"))
        return data
    else:
        raise Exception(f"Failed to fetch data from S3. Status code: {status}")


def save_data_to_s3(df, bucket_name, s3_key):
    """Save DataFrame to an S3 bucket."""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue())
    print(f"Successfully uploaded predictions to {bucket_name}/{s3_key}")


def load_model_from_mlflow(model_name, stage="Production"):
    """Load the latest model from MLFlow registry."""
    model_uri = f"models:/{model_name}/{stage}"
    print(f"Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def inference_pipeline():

    # Step 2: Load the latest model from MLFlow
    model_name = "StoreCasting"
    model_version_alias = 'approved'
    model_uri = f"models:/{model_name}@{model_version_alias}"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Step 1: Load data from S3
    #data = load_data_from_s3(bucket_name, input_s3_key)
    data = pd.read_csv('inputs/test.csv')
    X_test_final = data[[col for col in data.columns if col != 'sales']]

    forecast = pd.DataFrame({
        "date": data.date,
        "store": data.store,
        "item": data.item,
        "sales": model.predict(X_test_final)
    })

    # Step 4: Store the predictions back to S3
    prediction_df = pd.DataFrame(forecast, columns=["predictions"])
    #save_data_to_s3(prediction_df, bucket_name, output_s3_key)
    print(prediction_df)
    

if __name__ == "__main__":
    inference_pipeline()
