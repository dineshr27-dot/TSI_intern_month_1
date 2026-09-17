import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# 1. Load dataset
# -----------------------------
X, y = load_iris(return_X_y=True)

# -----------------------------
# 2. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 3. Model parameters
# -----------------------------
n_estimators = 100
max_depth = 10

# -----------------------------
# 4. Start MLflow run
# -----------------------------
with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # -----------------------------
    # 5. Make predictions
    # -----------------------------
    predictions = model.predict(X_test)

    # -----------------------------
    # 6. Calculate accuracy
    # -----------------------------
    accuracy = accuracy_score(y_test, predictions)

    # -----------------------------
    # 7. Log parameters
    # -----------------------------
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # -----------------------------
    # 8. Log metric
    # -----------------------------
    mlflow.log_metric("accuracy", accuracy)

    # -----------------------------
    # 9. Log model
    # -----------------------------
    mlflow.sklearn.log_model(
    model,
    name="random_forest_model",
    registered_model_name="RandomForestIris"
    )
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Tell Python to use the MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load dataset
X, y = load_iris(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model parameters
n_estimators = 100
max_depth = 10


# Start MLflow run
with mlflow.start_run():

    # Create model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Log and register model
    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        registered_model_name="RandomForestIris"
    )

    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"Accuracy: {accuracy:.4f}")