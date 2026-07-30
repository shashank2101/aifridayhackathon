"""
Trains a Random Forest classifier on the AI4I2020 predictive maintenance
dataset to predict machine failure, using both raw sensor features and
physics-based engineered features.

Usage:
    python3 train_model.py

Outputs:
    model/risk_model.joblib        - trained classifier
    model/feature_importance.json  - which parameters matter most
    Prints accuracy / precision / recall / confusion matrix / feature
    importance so you can show real numbers on demo day.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

# Raw sensor columns from AI4I2020
RAW_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

LABEL_COL = "Machine failure"  # 0 / 1


def engineer_features(df):
    """Add physics-based engineered features matching AI4I2020 failure rules.
    
    These features directly encode the physics behind the 5 failure modes,
    giving the model clear signals without having to learn the formulas
    from scratch."""
    df = df.copy()

    # Temperature differential → HDF trigger: fails when < 8.6K
    df["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]

    # Power in watts → PWF trigger: fails when < 3500W or > 9000W
    df["power_W"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * 2 * np.pi / 60

    # Overstrain product → OSF trigger: tool_wear × torque > threshold
    df["overstrain_product"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    # Torque-to-speed ratio — a general indicator of mechanical stress
    df["torque_speed_ratio"] = df["Torque [Nm]"] / (df["Rotational speed [rpm]"] + 1)

    return df


# All features used for training (raw + engineered)
ALL_FEATURES = RAW_FEATURES + [
    "temp_diff", "power_W", "overstrain_product", "torque_speed_ratio",
]


def main():
    df = pd.read_csv("data/ai4i2020.csv")
    print(f"Loaded {len(df)} rows from ai4i2020.csv")
    print(f"Failure rate: {df[LABEL_COL].mean() * 100:.1f}%")
    print(f"Failure breakdown:")
    for mode in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
        print(f"  {mode}: {df[mode].sum()}")
    print()

    # Engineer features
    df = engineer_features(df)

    X = df[ALL_FEATURES]
    y = df[LABEL_COL]

    # Stratified 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # handle minority class (failures)
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("=" * 55)
    print("VALIDATION RESULTS (20% held out from ai4i2020.csv)")
    print("=" * 55)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.3f}")
    print(f"F1 score : {f1_score(y_test, y_pred):.3f}")
    print()
    print("Confusion matrix (rows=actual, cols=predicted) [0=Normal, 1=Failure]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    # Feature importance
    importances = dict(zip(ALL_FEATURES, model.feature_importances_.round(4)))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))
    print("Feature importance (which parameters matter most):")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 50)
        print(f"  {feat:28s} {imp:.4f}  {bar}")

    # Save model + artifacts
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/risk_model.joblib")
    with open("model/feature_importance.json", "w") as f:
        json.dump(importances, f, indent=2)

    # Save metrics for the dashboard
    cm = confusion_matrix(y_test, y_pred).tolist()
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": cm,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "n_features": len(ALL_FEATURES),
        "feature_importance": importances,
    }
    with open("model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print()
    print("✓ Saved model to model/risk_model.joblib")
    print("✓ Saved feature importance to model/feature_importance.json")
    print("✓ Saved validation metrics to model/metrics.json")


if __name__ == "__main__":
    main()
