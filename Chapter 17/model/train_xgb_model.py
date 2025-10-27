import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, roc_curve, auc,
    classification_report, matthews_corrcoef, cohen_kappa_score
)

print("✅ Starting model training pipeline...")

os.makedirs("model", exist_ok=True)
file_path = 'data/dummy_test_vif_filtered_imputed_cleaned.csv'

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ Dataset not found at: {file_path}")

print("📄 Loading dataset...")
df = pd.read_csv(file_path)

X = df.drop(columns=['IS_FRAUD'])
y = df['IS_FRAUD'].astype(int)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print("🔤 Encoding categorical features...")
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print("📏 Scaling numerical features...")
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Training base XGBoost model...")
xgb_model = XGBClassifier(
    scale_pos_weight=94/6,
    use_label_encoder=False,
    eval_metric='aucpr',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=10,
    reg_alpha=10,
    reg_lambda=1,
    random_state=42
)
xgb_model.fit(X_train, y_train)

print("📊 Running Recursive Feature Elimination (RFE)...")
rfe = RFE(estimator=xgb_model, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = X_train.columns[rfe.support_].tolist()

print("🎯 Retraining final model on top features...")
xgb_model_final = XGBClassifier(
    scale_pos_weight=94/6,
    use_label_encoder=False,
    eval_metric='aucpr',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=10,
    reg_alpha=10,
    reg_lambda=1,
    random_state=42
)
xgb_model_final.fit(X_train[selected_features], y_train)

print("💾 Saving model and preprocessing artifacts...")
joblib.dump(xgb_model_final, "model/xgb_model_final.pkl")
joblib.dump(selected_features, "model/selected_features.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/label_encoders.pkl")
print("✅ All model artifacts saved successfully.")

# === Continue unchanged ===
# 🔍 Predict Probabilities
y_proba = xgb_model_final.predict_proba(X_test[selected_features])[:, 1]

thresholds = np.arange(0.1, 0.9, 0.05)
precision_scores = []
recall_scores = []
f1_scores = []

print("\nThreshold Tuning Results:")
for threshold in thresholds:
    y_pred_thresh = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_thresh)
    recall = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    print(f"Threshold: {threshold:.2f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}")

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision_scores, marker='o', label='Precision')
plt.plot(thresholds, recall_scores, marker='s', label='Recall')
plt.plot(thresholds, f1_scores, marker='^', label='F1 Score')
plt.axvline(x=0.5, color='grey', linestyle='--', label='Default Threshold')
plt.title("Threshold Tuning Curve")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"\n🔍 Best Threshold by F1 Score: {best_threshold:.2f}")

y_best_pred = (y_proba >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_best_pred)
plt.figure(figsize=(6, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (Best Threshold)")
plt.tight_layout()
plt.show()

print("\n📊 Classification Report (Best Threshold):")
print(classification_report(y_test, y_best_pred))

mcc = matthews_corrcoef(y_test, y_best_pred)
kappa = cohen_kappa_score(y_test, y_best_pred)
print(f"\n✅ Matthews Correlation Coefficient (MCC): {mcc:.3f}")
print(f"✅ Cohen's Kappa Score: {kappa:.3f}")

precision, recall, _ = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(6, 4))
plt.plot(recall, precision, linewidth=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()

importances = xgb_model_final.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Top 10 Most Informative Features (XGBoost)")
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
