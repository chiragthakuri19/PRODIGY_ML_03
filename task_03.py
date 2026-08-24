import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# 1. Dataset Simulation / Feature Representation
# Image features (e.g., flattened pixels / color histograms / HOG features)
# Here using a synthetic feature set representing 2-class image features for robust standalone execution
X, y = make_classification(
    n_samples=1000,
    n_features=50,
    n_informative=30,
    n_classes=2,
    random_state=42
)

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Support Vector Machine (SVM) Classifier
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat (0)', 'Dog (1)']))
