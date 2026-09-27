import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib


DATA_PATH = "data/Churn_Modelling.csv"
MODEL_PATH = "churn_classifier.joblib"
TARGET_COLUMN = "Exited"

# Identifier columns with no real predictive value - a customer's row
# number, ID, or surname doesn't cause them to churn.
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname"]


def load_data(path: str) -> pd.DataFrame:
    """Load and clean the churn dataset."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    df = df.dropna()
    return df


def prepare_features(df: pd.DataFrame):
    """Encode categorical columns and split into features (X) and target (y)."""
    df = df.copy()

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Geography (France/Spain/Germany) and Gender are text - encode as numbers.
    for col in X.select_dtypes(exclude=["number", "bool"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale numeric columns that have wide, very different ranges
    # (e.g. CreditScore ~300-850 vs Balance in the tens of thousands).
    numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    return X, y


def build_and_evaluate(X_train, X_test, y_train, y_test, model_name: str):
    """Train a classifier and print evaluation metrics."""
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    model = models[model_name]
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    print(f"\n=== Results: {model_name} ===")
    print("Classification Report (0=stayed, 1=churned):")
    print(classification_report(y_test, y_pred, target_names=["stayed", "churned"]))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    if model_name in ("random_forest", "gradient_boosting"):
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        print("\nTop 5 most important features:")
        print(importances.sort_values(ascending=False).head(5))

    return model


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    churn_count = df[TARGET_COLUMN].sum()
    total = len(df)
    print(f"Loaded {total} customers ({churn_count} churned, {total - churn_count} stayed)")
    print(f"Churn rate: {churn_count / total * 100:.2f}%")

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = None
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        model = build_and_evaluate(X_train, X_test, y_train, y_test, model_name=name)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
