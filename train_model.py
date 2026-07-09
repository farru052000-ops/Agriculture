import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = pd.read_csv("dataset/Crop_recommendation.csv")

# Display first 5 rows
print("First 5 Rows:")
print(data.head())

# Dataset Shape
print("\nDataset Shape:")
print(data.shape)

# Dataset Information
print("\nDataset Information:")
print(data.info())

# Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(data.describe())

# Crop Counts
print("\nCrop Counts:")
print(data["label"].value_counts())
# Correlation Heatmap

plt.figure(figsize=(10,8))
sns.heatmap(data.drop('label', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# Distribution Plots

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for feature in features:
    plt.figure(figsize=(6,4))
    sns.histplot(data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.show()
    # Boxplots

for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=data[feature])
    plt.title(f"Boxplot of {feature}")
    plt.show()
    # ======================================================
# Data Preprocessing
# ======================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Features and Target
X = data.drop('label', axis=1)
y = data['label']

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("\ny_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData preprocessing completed successfully!")
# ======================================================
# Machine Learning Models
# ======================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Naive Bayes": GaussianNB()
}

best_model = None
best_accuracy = 0

print("\n==============================")
print("Model Accuracies")
print("==============================")

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"{name}: {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

print("\nBest Model Accuracy:", best_accuracy)

# ======================================================
# Classification Report
# ======================================================

predictions = best_model.predict(X_test)

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))
# ======================================================
# Save Model and Scaler
# ======================================================

import os
import joblib

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save best model
joblib.dump(best_model, "model/model.pkl")

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

print("\nModel and scaler saved successfully!")