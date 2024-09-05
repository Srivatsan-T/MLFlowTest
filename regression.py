import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set the MLflow tracking URI (optional, if using a remote server)
mlflow.set_tracking_uri("http://localhost:5000")  # Update with your MLflow tracking server URI if not local
mlflow.set_experiment("Cali-Housing")

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Gradually improve the model by lowering the regularization strength (alpha)
for alpha in [10.0, 1.0]:
    with mlflow.start_run(run_name=f"Ridge_Regression_Improved_alpha_{alpha}"):
        # Initialize and train the model
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Log model parameters
        mlflow.log_param("alpha", alpha)

        # Calculate and log evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Log the model and get the ModelInfo object
        model_info = mlflow.sklearn.log_model(model, "ridge_model")
        registered_model_name = "RidgeRegression-Cali"
        # Register the model using the model_uri from ModelInfo
        mlflow.register_model(model_uri=model_info.model_uri, name=registered_model_name)

        print(f"Logged and registered model with alpha={alpha} | MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
