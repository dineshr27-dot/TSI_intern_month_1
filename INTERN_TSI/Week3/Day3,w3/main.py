import pandas as pd

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Data splitting and Cross Validation
from sklearn.model_selection import train_test_split, cross_val_score

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# 1. LOAD DATASET

data = pd.read_csv("placement.csv")

print("\nFirst 5 rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

# 2. SELECT FEATURES AND TARGET

# Input features
X = data[[
    "cgpa",
    "placement_exam_marks"
]]

# Target
y = data["placed"]

# 3. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 4. CREATE MODELS

logistic_model = LogisticRegression()

tree_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# 5. TRAIN MODELS

logistic_model.fit(X_train, y_train)

tree_model.fit(X_train, y_train)

forest_model.fit(X_train, y_train)

# 6. MAKE PREDICTIONS

logistic_pred = logistic_model.predict(X_test)

tree_pred = tree_model.predict(X_test)

forest_pred = forest_model.predict(X_test)

# 7. EVALUATION FUNCTION

def evaluate_model(model_name, y_actual, y_prediction):

    print(model_name)

    print(
        "Accuracy:",
        accuracy_score(y_actual, y_prediction)
    )

    print(
        "Precision:",
        precision_score(
            y_actual,
            y_prediction,
        )
    )

    print(
        "Recall:",
        recall_score(
            y_actual,
            y_prediction,
            zero_division=0
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_actual,
            y_prediction,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_actual,
            y_prediction
        )
    )


# 8. EVALUATE ALL MODELS

evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred
)

evaluate_model(
    "Decision Tree",
    y_test,
    tree_pred
)

evaluate_model(
    "Random Forest",
    y_test,
    forest_pred
)

# 9. CROSS VALIDATION

print("5-FOLD CROSS VALIDATION")

models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": tree_model,
    "Random Forest": forest_model
}


for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="accuracy"
    )

    print("\n", name)

    print(
        "Cross Validation Scores:",
        scores
    )

    print(
        "Average CV Accuracy:",
        scores.mean()
    )


# 10. USER INPUT

print("STUDENT PLACEMENT PREDICTION")


cgpa = float(
    input("Enter CGPA: ")
)

exam_marks = float(
    input("Enter Placement Exam Marks: ")
)


# Create DataFrame with same columns
# used during model training

new_student = pd.DataFrame(
    [{
        "cgpa": cgpa,
        "placement_exam_marks": exam_marks
    }]
)


# 11. PREDICT USER INPUT

prediction = forest_model.predict(
    new_student
)


# 12. DISPLAY RESULT

if prediction[0] == 1:

    print(
        "\nPrediction: Student may be PLACED"
    )

else:

    print(
        "\nPrediction: Student may NOT be PLACED"
    )