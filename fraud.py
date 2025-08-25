# =========================
# 0) Imports & Config
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (roc_auc_score, precision_recall_curve, auc,
                             classification_report, confusion_matrix, RocCurveDisplay,
                             PrecisionRecallDisplay)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.inspection import permutation_importance
import shap
import warnings
warnings.filterwarnings('ignore')

# OPTIONAL: Set a global random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# 1) Load / Assume df is present
# =========================
# df = pd.read_csv("your_transactions.csv")  # <- if needed

# Sanity check for the expected columns (adjust if your frame differs)
required_cols = {
    'transaction_id','timestamp','sender_account','receiver_account','transaction_type',
    'merchant_category','location','device_used','is_fraud','fraud_type',
    'time_since_last_transaction','spending_deviation_score','velocity_score',
    'geo_anomaly_score','payment_channel','ip_address','device_hash','amount_ngn',
    'bvn_linked','new_device_transaction','sender_persona'
}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# =========================
# 2) Cleaning & Feature Engineering
# =========================
df = df.copy()

# 2.1 Time features
# Convert timestamp to datetime (if not already)
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# 2.2 Transform amounts (log1p to reduce skew)
df['amount_log'] = np.log1p(df['amount_ngn'])

# 2.3 Booleans -> ints
df['bvn_linked'] = df['bvn_linked'].astype(int)
df['new_device_transaction'] = df['new_device_transaction'].astype(int)

# 2.4 Drop high-cardinality identifiers & target-leaking or unused columns
drop_cols = [
    'transaction_id', 'sender_account', 'receiver_account',
    'ip_address', 'device_hash',
    'fraud_type',           # mostly missing in your sample; not a reliable feature
    'timestamp',            # replaced by hour/day/is_weekend
    'amount_ngn'            # replaced by amount_log
]
df = df.drop(columns=drop_cols)

# 2.5 Target & features
target = 'is_fraud'
y = df[target].astype(int)
X = df.drop(columns=[target])

# 2.6 Define categorical / numeric columns
cat_cols_low_card = [
    'transaction_type', 'merchant_category', 'location',
    'device_used', 'payment_channel', 'sender_persona'
]
# Only keep those that actually exist (safety)
cat_cols_low_card = [c for c in cat_cols_low_card if c in X.columns]

num_cols = [c for c in X.columns if c not in cat_cols_low_card]

# =========================
# 3) Train/Validation Split (Stratified)
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# =========================
# 4) Column Transformer (Impute + Encode + Scale)
# =========================
numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=False))  # with_mean=False is sparse-friendly
])

cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", cat_tf, cat_cols_low_card),
    ],
    remainder='drop'
)

# =========================
# 5) Model (RandomForest) + Class imbalance handling
# =========================
# Approach: use class_weight on the model for imbalance (memory safe)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    class_weight='balanced_subsample',
    random_state=RANDOM_STATE
)

pipe = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", rf)
])

# =========================
# 6) Hyperparameter Search (RandomizedSearchCV)
# =========================
param_distributions = {
    "clf__n_estimators": [200, 300, 400],
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5, 0.8]
}

search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_distributions,
    n_iter=15,
    scoring="average_precision",  # PR-AUC is better for imbalanced data
    cv=3,
    verbose=1,
    n_jobs=-1,
    random_state=RANDOM_STATE
)

# Weighted samples (alternative to class_weight). Here we keep class_weight.
# If you want sample weights instead:
# w_train = compute_sample_weight(class_weight='balanced', y=y_train)
# search.fit(X_train, y_train, clf__sample_weight=w_train)

search.fit(X_train, y_train)

print("Best params:", search.best_params_)
print("Best CV PR-AUC:", search.best_score_)

best_model = search.best_estimator_

# =========================
# 7) Evaluation on Validation Set
# =========================
y_proba = best_model.predict_proba(X_valid)[:, 1]
y_pred_default = (y_proba >= 0.5).astype(int)

roc = roc_auc_score(y_valid, y_proba)
precision, recall, thresholds = precision_recall_curve(y_valid, y_proba)
pr_auc = auc(recall, precision)
print(f"Validation ROC-AUC: {roc:.4f}")
print(f"Validation PR-AUC:  {pr_auc:.4f}")
print("\nClassification report @0.50 threshold:\n",
      classification_report(y_valid, y_pred_default, digits=4))

cm = confusion_matrix(y_valid, y_pred_default)
print("Confusion matrix @0.50:\n", cm)

# =========================
# 8) Threshold Tuning (maximize F1 on PR curve)
# =========================
f1_scores = (2*precision*recall) / (precision + recall + 1e-12)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
print(f"Best threshold by F1: {best_threshold:.4f}, F1={f1_scores[best_idx]:.4f}")

y_pred_opt = (y_proba >= best_threshold).astype(int)
print("\nClassification report @optimal threshold:\n",
      classification_report(y_valid, y_pred_opt, digits=4))
print("Confusion matrix @optimal threshold:\n",
      confusion_matrix(y_valid, y_pred_opt))

# =========================
# 9) Plots (ROC, PR, Confusion Matrix, Feature Importance)
# =========================
plt.figure()
RocCurveDisplay.from_predictions(y_valid, y_proba)
plt.title("ROC Curve (Validation)")
plt.show()

plt.figure()
PrecisionRecallDisplay(precision=precision, recall=recall).plot()
plt.title(f"Precision-Recall (PR-AUC={pr_auc:.4f})")
plt.show()

# Confusion matrix heatmap
def plot_cm(cm, title="Confusion Matrix", labels=("Not Fraud", "Fraud")):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label',
           title=title)
    # annotate
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.show()

plot_cm(cm, title="Confusion Matrix @0.50")

# Permutation importance on validation (slower but model-agnostic, gives post-encoding importance)
print("\nComputing permutation importances (this can take a while)...")
perm = permutation_importance(best_model, X_valid, y_valid, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1)
# Get feature names after preprocessing
ohe = best_model.named_steps['prep'].named_transformers_['cat'].named_steps['ohe']
cat_feature_names = []
if len(cat_cols_low_card) > 0:
    cat_feature_names = ohe.get_feature_names_out(cat_cols_low_card).tolist()
feature_names = num_cols + cat_feature_names

pi = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
print("\nTop 20 permutation importances:\n", pi.head(20))

# Plot top 20
pi.head(20).iloc[::-1].plot(kind='barh', figsize=(8,6))
plt.title("Top 20 Feature Importances (Permutation, Validation)")
plt.xlabel("Importance (mean decrease in score)")
plt.tight_layout()
plt.show()

# =========================
# 10) Explainability with SHAP
# =========================
# Use a small sample for SHAP (for speed/memory)
shap_sample_size = min(20000, X_valid.shape[0])
X_valid_sample = X_valid.sample(shap_sample_size, random_state=RANDOM_STATE)
y_valid_sample = y_valid.loc[X_valid_sample.index]

# Get the fitted RF inside the pipeline
rf_fitted = best_model.named_steps['clf']
preprocessor = best_model.named_steps['prep']

# Transform sample to model-ready matrix and corresponding feature names
X_valid_enc = preprocessor.transform(X_valid_sample)

# SHAP expects dense for some plots; convert safely if not too big
try:
    X_valid_enc_dense = X_valid_enc.toarray()
except:
    X_valid_enc_dense = X_valid_enc  # fallback if already dense

explainer = shap.TreeExplainer(rf_fitted)
shap_values = explainer.shap_values(X_valid_enc_dense)

# SHAP summary (class 1 = fraud)
# Map encoded cols back to names
encoded_feature_names = feature_names

# Global explanation
shap.summary_plot(shap_values[1], X_valid_enc_dense, feature_names=encoded_feature_names, show=True)

# Local explanation (pick a high-score example)
ix = np.argsort(y_proba[X_valid_sample.index])[-1]  # index of a likely fraud
row = X_valid_enc_dense[ix]
shap.force_plot(explainer.expected_value[1], shap_values[1][ix], features=row,
                feature_names=encoded_feature_names, matplotlib=True)
plt.show()

# =========================
# 11) OPTIONAL: Memory-safe SMOTE on a small training subset
# (Use only if you specifically want oversampling—class_weight is usually enough at this scale.)
# =========================
# from imblearn.over_sampling import SMOTE
# from sklearn.linear_model import LogisticRegression
# # Sample down the majority class to make SMOTE feasible
# sub_size = 600_000
# X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=sub_size,
#                                       stratify=y_train, random_state=RANDOM_STATE)
# X_sub_enc = preprocessor.fit_transform(X_sub)  # fit on the subset
# smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.2, k_neighbors=3, n_jobs=-1)
# X_sm, y_sm = smote.fit_resample(X_sub_enc, y_sub)
# lr = LogisticRegression(max_iter=200, n_jobs=-1, class_weight=None)
# lr.fit(X_sm, y_sm)
# # Evaluate on full validation
# y_val_proba_lr = lr.predict_proba(preprocessor.transform(X_valid))[:,1]
# print("SMOTE+LR PR-AUC:", auc(*precision_recall_curve(y_valid, y_val_proba_lr)[::-1]))
