"""
train_model.py
----------------
Rebuilds the Employee Attrition Decision Tree model from the notebook
in one clean, linear run (no out-of-order cell issues) and saves
everything the Streamlit app needs into a single model_bundle.pkl file.

Run this once from the terminal:
    python train_model.py
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
CSV_PATH = "HR_Employee_Attrition.csv"
df = pd.read_csv(CSV_PATH, encoding="cp1252")

# ----------------------------------------------------------------------
# 2. Select features and target (same columns as your notebook)
# ----------------------------------------------------------------------
CATEGORICAL_COLS = ["BusinessTravel", "Department", "EducationField"]
NUMERIC_COLS = ["Age", "DailyRate", "DistanceFromHome", "Education"]

X = df.drop(columns=["Attrition", "EmployeeCount", "EmployeeNumber"])
y = df["Attrition"].map({"Yes": 1, "No": 0})

# Save the exact category options seen during training.
# The app will use these to build its dropdown menus, so they can
# never be typed incorrectly and can never fall out of sync with the model.
category_options = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}

# ----------------------------------------------------------------------
# 3. One-hot encode categorical columns (same as pd.get_dummies in notebook)
# ----------------------------------------------------------------------
X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
X = X.astype(float)

# The exact column order the model was trained on -> app must recreate this
feature_columns = X.columns.tolist()

# ----------------------------------------------------------------------
# 4. Train / test split
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------------------------------------------------
# 5. Train the Decision Tree (this was the best-performing model
#    in your notebook comparison, by F1 score / recall)
# ----------------------------------------------------------------------
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 6. Evaluate
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)
print("Model performance on test set:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.3f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.3f}")

# ----------------------------------------------------------------------
# 7. Save everything the app needs in ONE file
# ----------------------------------------------------------------------
bundle = {
    "model": model,
    "feature_columns": feature_columns,
    "numeric_cols": NUMERIC_COLS,
    "categorical_cols": CATEGORICAL_COLS,
    "category_options": category_options,
}
joblib.dump(bundle, "model_bundle.pkl")
print("\nSaved model_bundle.pkl — you're ready to run the Streamlit app.")
