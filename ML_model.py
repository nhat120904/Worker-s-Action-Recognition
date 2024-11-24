import numpy as np
from dataset import ActivityDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

dataset = ActivityDataset(root="./raw_data")

# Flatten each sample in the dataset
X = [sample.to_numpy().flatten() for sample in dataset.data]  # Features
y = dataset.label  # Labels

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Random Forest Model ---
print("Training Random Forest Model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate Random Forest
y_pred_rf = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, y_pred_rf)
print(f"Random Forest Validation Accuracy: {rf_accuracy:.4f}")
print("Random Forest Classification Report:")
print(classification_report(y_val, y_pred_rf))

# Save the Random Forest model
joblib.dump(rf_model, 'RF_raw_knife.pkl')

# --- SVM Model ---
print("\nTraining SVM Model...")
svm_model = SVC(kernel='rbf', C=1, random_state=42)  # You can adjust kernel and C for tuning
svm_model.fit(X_train, y_train)

# Predict and evaluate SVM
y_pred_svm = svm_model.predict(X_val)
svm_accuracy = accuracy_score(y_val, y_pred_svm)
print(f"SVM Validation Accuracy: {svm_accuracy:.4f}")
print("SVM Classification Report:")
print(classification_report(y_val, y_pred_svm))

# Save the SVM model
joblib.dump(svm_model, 'SVM_raw_knife.pkl')