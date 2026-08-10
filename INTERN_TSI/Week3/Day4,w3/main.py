import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. LOAD DATASET

data = pd.read_csv(
    "ecommerceDataset.csv",
    header=None,
    names=["category", "description"]
)

# 2. CLEAN DATA

data = data.dropna(
    subset=["category", "description"]
)

data = data.drop_duplicates()

# 3. FEATURES AND TARGET

X = data["description"]
y = data["category"]

# 4. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. TF-IDF + RANDOM FOREST PIPELINE

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=40000,
            ngram_range=(1, 2)
        )
    ),

    (
        "random_forest",
        RandomForestClassifier(
            n_estimators=350,
            random_state=42,
            n_jobs=-1
        )
    )
])

# 6. TRAIN MODEL

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# 7. PREDICT TEST DATA

y_pred = model.predict(
    X_test
)

# 8. EVALUATE MODEL

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

print("\n===== MODEL PERFORMANCE =====")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 9. CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\n===== CONFUSION MATRIX =====")

print(cm)

# 10. CONFUSION MATRIX GRAPH

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 11. MODEL PERFORMANCE GRAPH

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

metric_scores = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_scores
)

plt.title("Model Performance")

plt.xlabel("Evaluation Metrics")

plt.ylabel("Score")

plt.ylim(0, 1)

# Display score above each bar

for bar, score in zip(bars, metric_scores):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{score:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show(block=False)

# 12. USER INPUT PREDICTION

print("\n===== PRODUCT CATEGORY PREDICTION =====")

user_text = input(
    "Enter product description: "
)

prediction = model.predict(
    [user_text]
)

print(
    "Predicted Category:",
    prediction[0]
)