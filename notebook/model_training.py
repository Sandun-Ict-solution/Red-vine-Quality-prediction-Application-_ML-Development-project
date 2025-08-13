import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import numpy as np

# ---------------------------
# 1. Download dataset automatically if missing
# ---------------------------
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
DATA_PATH = "winequality-red.csv"

if not os.path.exists(DATA_PATH):
    print("🔥 Downloading Wine Quality dataset...")
    df = pd.read_csv(DATA_URL, sep=';')
    df.to_csv(DATA_PATH, index=False)
else:
    print("✅ Using existing Wine Quality dataset...")
    df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ---------------------------
# 2. Data preprocessing
# ---------------------------
# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Convert wine quality to binary classification (good vs average/poor)
# Wines with quality >= 7 are considered "good" (1), others are "average/poor" (0)
df['quality_binary'] = (df['quality'] >= 7).astype(int)

print(f"Quality distribution:")
print(df['quality'].value_counts().sort_index())
print(f"Binary quality distribution:")
print(df['quality_binary'].value_counts())

# Features (all chemical properties) and target
feature_columns = [col for col in df.columns if col not in ['quality', 'quality_binary']]
X = df[feature_columns]
y = df['quality_binary']

print(f"Features: {feature_columns}")

# ---------------------------
# 3. Model training and comparison
# ---------------------------
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Define models to compare
models = {
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}

# Train and evaluate models
best_model = None
best_score = 0
model_results = {}

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test set evaluation
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    
    # Store results
    model_results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_score': test_score,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  Test accuracy: {test_score:.4f}")
    
    # Select best model based on test score
    if test_score > best_score:
        best_score = test_score
        best_model = model
        best_model_name = name

print(f"\n🏆 Best model: {best_model_name} (Test Accuracy: {best_score:.4f})")

# ---------------------------
# 4. Detailed evaluation of best model
# ---------------------------
print("\n" + "="*50)
print(f"DETAILED EVALUATION - {best_model_name}")
print("="*50)

y_pred_best = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['Average/Poor', 'Good']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Feature importance (if Random Forest is the best model)
if best_model_name == 'Random Forest':
    feature_importance = best_model.named_steps['classifier'].feature_importances_
    feature_names = feature_columns
    
    print("\nTop 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

# ---------------------------
# 5. Save the best model and preprocessing info
# ---------------------------
joblib.dump(best_model, "wine_model.pkl")
print(f"\n✅ Best model ({best_model_name}) saved as wine_model.pkl")

# Save feature names and other metadata
model_info = {
    'feature_columns': feature_columns,
    'model_name': best_model_name,
    'test_accuracy': best_score,
    'model_results': {k: {
        'cv_mean': v['cv_mean'],
        'cv_std': v['cv_std'], 
        'test_score': v['test_score']
    } for k, v in model_results.items()}
}

joblib.dump(model_info, "wine_model_info.pkl")
print("✅ Model information saved as wine_model_info.pkl")

print(f"\n📊 Final Test Accuracy: {best_score:.4f}")
print("🍷 Wine Quality Model Training Complete!")