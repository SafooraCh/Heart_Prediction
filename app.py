# =============================================================================
# LAB 08: SUPPORT VECTOR MACHINE - HEART FAILURE DATASET (Complete Lab Activity)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification # Used for data simulation
import pickle
import warnings
import os

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*75)
print("LAB 08: SUPPORT VECTOR MACHINE - HEART FAILURE CLASSIFICATION")
print("="*75)

# =============================================================================
# STEP 1: LOAD DATASET (SIMULATED FOR STANDALONE EXECUTION)
# =============================================================================
print("\n📁 SIMULATING DATASET...")
print("-"*75)

# Define feature names matching the original Heart Failure dataset
FEATURE_NAMES = [
    'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
    'ejection_fraction', 'high_blood_pressure', 'platelets',
    'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
]
TARGET_NAME = 'DEATH_EVENT'

# Generate synthetic data with similar dimensions and characteristics
X_synth, y_synth = make_classification(
    n_samples=299, # Original sample count
    n_features=12,
    n_informative=8,
    n_redundant=0,
    n_classes=2,
    weights=[0.68, 0.32], # Approximate class balance
    flip_y=0.05,
    random_state=42
)

# Create the DataFrame
df = pd.DataFrame(X_synth, columns=FEATURE_NAMES)
df[TARGET_NAME] = y_synth.astype(int)

print("✓ Synthetic Dataset loaded successfully!")
print("NOTE: Original data replaced with synthetic data for portability.")
print("\nDataset Shape:", df.shape)
print("Number of Samples:", df.shape[0])
print("Number of Features:", df.shape[1] - 1)

# =============================================================================
# STEP 2: DATASET EXPLORATION
# =============================================================================
print("\n" + "="*75)
print("📊 DATASET EXPLORATION")
print("="*75)

print("\n1️⃣ First 5 rows:")
print(df.head())

print("\n2️⃣ Dataset Information:")
# Since the data is synthetic, .info() is less useful, but kept for structure
print(df.info(verbose=False))

print("\n3️⃣ Missing Values:")
print(df.isnull().sum())

print("\n4️⃣ Statistical Summary:")
print(df.describe())

print("\n5️⃣ Target Distribution (DEATH_EVENT):")
print(df[TARGET_NAME].value_counts())

# Bar chart
plt.figure(figsize=(6, 4))
df[TARGET_NAME].value_counts().plot(kind='bar', color=['#4ECDC4', '#FF6B6B'])
plt.title('Death Event Distribution (Synthetic Data)')
plt.xlabel('DEATH_EVENT')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('death_event_distribution.png')
# plt.show() # Uncomment to display plot
plt.close() # Close figure to free memory


# =============================================================================
# STEP 3: DATA VISUALIZATION
# =============================================================================
print("\n" + "="*75)
print("📈 DATA VISUALIZATION")
print("="*75)

# Correlation heatmap
print("Generating correlation heatmap (correlation_heatmap.png)...")
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Synthetic Data)")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
# plt.show() # Uncomment to display plot
plt.close() # Close figure to free memory


# =============================================================================
# STEP 4: DATA PREPROCESSING
# =============================================================================
print("\n" + "="*75)
print("🔧 DATA PREPROCESSING")
print("="*75)

X = df.drop(TARGET_NAME, axis=1)
y = df[TARGET_NAME]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📊 Data Split:")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =============================================================================
# STEP 5: TRAINING SVM MODEL
# =============================================================================
print("\n" + "="*75)
print("🤖 TRAINING SVM MODEL")
print("="*75)

print("\nTraining SVM (RBF kernel)...")
# Note: For real-world data, scaling the features (e.g., StandardScaler) is highly
# recommended before training an SVM, especially with the RBF kernel.
model = SVC(kernel='rbf', gamma='auto', random_state=42)
model.fit(X_train, y_train)
print("✓ Model training completed!")

# Predictions
y_pred = model.predict(X_test)


# =============================================================================
# STEP 6: EVALUATION
# =============================================================================
print("\n" + "="*75)
print("📊 MODEL EVALUATION")
print("="*75)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy (RBF): {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\n📋 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Plot Confusion Matrix
print("Generating confusion matrix plot (confusion_matrix.png)...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Survived', 'Died'],
            yticklabels=['Survived', 'Died'])
plt.title("Confusion Matrix - SVM Model (RBF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
# plt.show() # Uncomment to display plot
plt.close() # Close figure to free memory

print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Survived', 'Died']))


# =============================================================================
# STEP 7: KERNEL COMPARISON
# =============================================================================
print("\n" + "="*75)
print("🔬 COMPARING DIFFERENT SVM KERNELS")
print("="*75)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_results = {}

for k in kernels:
    print(f"Training kernel: {k}")
    m = SVC(kernel=k, gamma='auto', random_state=42)
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    acc = accuracy_score(y_test, pred)
    kernel_results[k] = acc
    print(f" → Accuracy = {acc:.4f}")

# Plot comparison
print("Generating kernel comparison plot (kernel_comparison.png)...")
plt.figure(figsize=(8, 5))
plt.bar(kernel_results.keys(), kernel_results.values(),
        color=['#FF6B6B','#4ECDC4','#45B7D1','#FFA07A'])
plt.title("Kernel Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim([min(kernel_results.values())*0.95, 1.05])
plt.tight_layout()
plt.savefig("kernel_comparison.png")
# plt.show() # Uncomment to display plot
plt.close() # Close figure to free memory

best_kernel = max(kernel_results, key=kernel_results.get)
print(f"\n🏆 Best Kernel: {best_kernel.upper()} with accuracy {kernel_results[best_kernel]:.4f}")


# =============================================================================
# STEP 8: SAVE MODEL
# =============================================================================
print("\n" + "="*75)
print("💾 SAVING MODEL")
print("="*75)

try:
    with open("svm_heart_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✓ Model saved as 'svm_heart_model.pkl'")
except Exception as e:
    print(f"Could not save model: {e}")


# =============================================================================
# STEP 9: SAMPLE PREDICTION
# =============================================================================
print("\n" + "="*75)
print("🧪 SAMPLE PREDICTION")
print("="*75)

# Select 3 samples from the test set for a real-world prediction scenario
sample = X_test.sample(3, random_state=1)
print("\nSample Data (Features):")
print(sample)

sample_pred_raw = model.predict(sample)
sample_predictions = ['Died' if p == 1 else 'Survived' for p in sample_pred_raw]
print("\nPredictions (RBF Model):", sample_predictions)


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*75)
print("📋 FINAL SUMMARY")
print("="*75)

print(f"""
Dataset: Heart Failure Clinical Records (Simulated)
Total Samples: {df.shape[0]}
Features: {df.shape[1]-1}
Target: DEATH_EVENT

Accuracy (RBF kernel): {accuracy:.4f}
Best Kernel: {best_kernel.upper()}

Files Generated:
    - svm_heart_model.pkl (Model artifact)
    - confusion_matrix.png (Evaluation plot)
    - kernel_comparison.png (Comparison plot)
    - correlation_heatmap.png (Exploration plot)
""")
