# %% [markdown]
# # Typhoid Prediction - Complete Implementation with Hyperparameter Tuning
# ## Using Clinical and Widal Test Data
#
# **Research Project:** Predictive Diagnostic Model for Typhoid using Stacked Ensemble Machine Learning

# %% [markdown]
# ## 1. Installation and Imports

# %%
# Install required packages
!pip install imbalanced-learn xgboost

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  # <-- THIS IS THE MISSING LINE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import time
import warnings
import re
warnings.filterwarnings('ignore')

print("✅ All packages imported successfully!")

# %% [markdown]
# ## 2. Data Loading from Google Drive

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Load your actual typhoid dataset
def load_typhoid_data():
    """Load your typhoid clinical dataset from Google Drive"""
    try:
        # IMPORTANT: Update this path to your actual typhoid file location
        file_path = "/content/drive/MyDrive/disease_dataset/typhoid_data.csv" #<-- MAKE SURE THIS FILENAME IS CORRECT
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {len(df)} records")

        # Display basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"🎯 Target distribution (before cleaning):")
        print(df['Final Output'].value_counts())

        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("Please check the file path and try again.")
        return None

# Load the data
df = load_typhoid_data()

if df is not None:
    display(df.head())
    print("\n📋 Column names:")
    print(df.columns.tolist())

# %% [markdown]
# ## 3. Data Preprocessing and Feature Engineering

# %%
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def _engineer_widal_features(self, df):
        """Extracts numerical features from the 'Widal Test' column."""
        print("... Engineering features from 'Widal Test' column ...")

        def extract_dilution(text, antigen):
            if not isinstance(text, str): return 0
            match = re.search(f"{antigen}\\s*=\\s*1:(\\d+)", text, re.IGNORECASE)
            return float(match.group(1)) if match else 0

        df['Widal_TO'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'TO'))
        df['Widal_TH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'TH'))
        df['Widal_AH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'AH'))
        df['Widal_BH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'BH'))

        df = df.drop(columns=['Widal Test'])
        return df

    def preprocess_data(self, df, target_col='Final Output', test_size=0.2, augment_data=True):
        """Preprocess and optionally augment the typhoid data"""
        print("🔄 Preprocessing data...")
        df_processed = df.copy()

        # --- Feature Engineering ---
        df_processed = self._engineer_widal_features(df_processed)

        # --- Data Cleaning and Encoding ---
        df_processed[target_col] = df_processed[target_col].str.strip().str.lower().map({'positive': 1, 'negative': 0})
        df_processed.dropna(subset=[target_col], inplace=True)
        df_processed[target_col] = df_processed[target_col].astype(int)

        # --- FIX #2: Clean whitespace from categorical columns before encoding ---
        categorical_cols = ['Blood Group', 'ESR']
        for col in categorical_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].str.strip()

        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)

        if 'Gender' in df_processed.columns:
            df_processed['Gender'] = LabelEncoder().fit_transform(df_processed['Gender'])

        for col in df_processed.select_dtypes(include=np.number).columns:
            if df_processed[col].isnull().sum() > 0:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # --- FIX #1: REMOVE THE DATA LEAKAGE SOURCE ---
        print("... Removing 'Blood Culture' from features to prevent data leakage ...")
        if 'Blood Culture' in df_processed.columns:
            df_processed = df_processed.drop(columns=['Blood Culture'])

        # --- Final Preparation ---
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        self.feature_names = X.columns.tolist()

        print(f"Original class distribution: {y.value_counts().to_dict()}")

        if augment_data:
            X, y = self._augment_data(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

        numerical_cols_to_scale = [col for col in ['Age', 'WBC Count', 'Widal_TO', 'Widal_TH', 'Widal_AH', 'Widal_BH'] if col in X_train.columns]
        scaler = StandardScaler()
        X_train[numerical_cols_to_scale] = scaler.fit_transform(X_train[numerical_cols_to_scale])
        X_test[numerical_cols_to_scale] = scaler.transform(X_test[numerical_cols_to_scale])

        print(f"✅ Final dataset: {len(X_train)} training, {len(X_test)} test samples")
        print(f"📊 Training class distribution: {y_train.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test

    def _augment_data(self, X, y):
        """Augment data using SMOTE"""
        print("🔄 Augmenting data with SMOTE...")
        original_counts = y.value_counts()
        k_neighbors = min(5, original_counts.min() - 1) if original_counts.min() > 1 else 1

        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        print(f"✅ After augmentation: {len(X_resampled)} samples")
        print(f"📊 New class distribution: {y_resampled.value_counts().to_dict()}")

        return X_resampled, y_resampled
# %%
# Preprocess the data
preprocessor = DataPreprocessor()
X_train, X_test, y_train, y_test = preprocessor.preprocess_data(df, augment_data=True)

print(f"\n🔍 Features used for modeling:")
for i, feature in enumerate(preprocessor.feature_names, 1):
    print(f"{i:2d}. {feature}")

# %% [markdown]
# ## 4. Stacked Ensemble & Hyperparameter Tuning
# The structure for training, tuning, and evaluation remains the same as the previous scripts.

# %%
# The ClinicalStackedEnsemble class is identical to the one in the malaria script.
class ClinicalStackedEnsemble:
    def __init__(self):
        self.base_models = []
        self.meta_learner = None
        self.cv_scores = {}
        self.feature_importance = None
        self.feature_names = None

    def initialize_base_models(self):
        self.base_models = [
            ('random_forest', RandomForestClassifier(random_state=42, n_jobs=-1)),
            ('xgboost', XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('svm', SVC(probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_jobs=-1)),
            ('logistic', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
        ]
        print("✅ Base models initialized:", [name for name, _ in self.base_models])

    def train_base_models(self, X_train, y_train):
        print("\n🔧 Training base models with cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in self.base_models:
            print(f"Training {name}...")
            start_time = time.time()
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_weighted')
            self.cv_scores[name] = {'mean': cv_scores.mean(), 'std': cv_scores.std()}
            print(f"  ✅ {name} trained in {time.time() - start_time:.2f}s | CV F1: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    def generate_meta_features(self, X, y):
        print("\n🔄 Generating meta-features...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.hstack([cross_val_predict(model, X, y, cv=skf, method='predict_proba') for _, model in self.base_models])
        print(f"✅ Meta-features shape: {meta_features.shape}")
        return meta_features

    def train_meta_learner(self, meta_features, y):
        print("\n🔧 Training meta-learner...")
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_learner.fit(meta_features, y)
        print(f"✅ Meta-learner training completed. Score: {self.meta_learner.score(meta_features, y):.4f}")

    def train_stacked_ensemble(self, X_train, y_train, feature_names):
        print("=" * 60, "\n🚀 TRAINING STACKED ENSEMBLE MODEL\n" + "=" * 60)
        start_time = time.time()
        self.feature_names = feature_names
        self.initialize_base_models()
        self.train_base_models(X_train, y_train)
        meta_features = self.generate_meta_features(X_train, y_train)
        self.train_meta_learner(meta_features, y_train)
        self._calculate_feature_importance()
        print(f"\n✅ Stacked ensemble training completed! Total time: {time.time() - start_time:.2f}s")

    def predict(self, X):
        base_predictions = np.hstack([model.predict_proba(X) for _, model in self.base_models])
        return self.meta_learner.predict(base_predictions)

    def predict_proba(self, X):
        base_predictions = np.hstack([model.predict_proba(X) for _, model in self.base_models])
        return self.meta_learner.predict_proba(base_predictions)

    def _calculate_feature_importance(self):
        rf_model = next((model for name, model in self.base_models if name == 'random_forest'), None)
        if rf_model and hasattr(self, 'feature_names'):
            self.feature_importance = pd.DataFrame({'feature': self.feature_names, 'importance': rf_model.feature_importances_}).sort_values('importance', ascending=False)

# Train the baseline ensemble
print("🎯 Starting Baseline Stacked Ensemble Training for Typhoid...")
ensemble = ClinicalStackedEnsemble()
ensemble.train_stacked_ensemble(X_train, y_train, preprocessor.feature_names)

# Hyperparameter Tuning
print("\n" + "="*60 + "\n⚙️  HYPERPARAMETER TUNING FOR RANDOM FOREST\n" + "="*60)
param_grid = {
    'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=50, cv=5, verbose=1, random_state=42, n_jobs=-1, scoring='f1_weighted')
print("Fitting RandomizedSearchCV...")
rf_random_search.fit(X_train, y_train)
print(f"\n✅ Tuning complete! Best Parameters: {rf_random_search.best_params_}")
best_rf_model = rf_random_search.best_estimator_

# %% [markdown]
# ## 5. Comprehensive Model Evaluation

# %%
# The ComprehensiveEvaluator class is also identical.
class ComprehensiveEvaluator:
    def __init__(self):
        self.results = {}
        self.metrics_history = []

    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        print(f"\n{'='*50}\n📊 EVALUATING: {model_name.upper()}\n{'='*50}")
        y_pred = model.predict(X_test)

        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            results['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            self._plot_comprehensive_evaluation(y_test, y_pred, y_proba, model_name, results['roc_auc'])
        else:
            results['roc_auc'] = 'N/A'

        self.metrics_history.append(results)
        self._print_detailed_results(results)
        return results

    def _print_detailed_results(self, results):
        print(f"🎯 PERFORMANCE METRICS:\n"
              f"   Accuracy:    {results['accuracy']:.4f}\n"
              f"   Precision:   {results['precision']:.4f}\n"
              f"   Recall:      {results['recall']:.4f}\n"
              f"   F1-Score:    {results['f1_score']:.4f}\n"
              f"   ROC AUC:     {results.get('roc_auc', 'N/A'):.4f}")

    def _plot_comprehensive_evaluation(self, y_test, y_pred, y_proba, model_name, roc_auc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Evaluation: {model_name}', fontsize=16)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', ax=axes[0], cmap='Blues')
        axes[0].set_title('Normalized Confusion Matrix')
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def compare_base_models(self, ensemble, X_test, y_test):
        print("\n" + "="*60 + "\n🔍 COMPARING BASELINE MODEL PERFORMANCE\n" + "="*60)
        return {name: self.evaluate_model(model, X_test, y_test, f"Base_{name}") for name, model in ensemble.base_models}

    def create_comparison_table(self):
        return pd.DataFrame(self.metrics_history).round(4)

# %%
# Run all evaluations
evaluator = ComprehensiveEvaluator()
tuned_rf_results = evaluator.evaluate_model(best_rf_model, X_test, y_test, "Tuned_Random_Forest")
ensemble_results = evaluator.evaluate_model(ensemble, X_test, y_test, "Stacked_Ensemble")
base_results = evaluator.compare_base_models(ensemble, X_test, y_test)

# %% [markdown]
# ## 6. Performance Comparison and Final Analysis

# %%
print("\n" + "="*70 + "\n🏆 FINAL PERFORMANCE COMPARISON\n" + "="*70)
comparison_df = evaluator.create_comparison_table().sort_values('f1_score', ascending=False)
print("\n📈 Model Performance Ranking (Sorted by F1-Score):")
display(comparison_df)

print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES (from Tuned Random Forest):")
tuned_feature_importance = pd.DataFrame({'feature': preprocessor.feature_names, 'importance': best_rf_model.feature_importances_}).sort_values('importance', ascending=False)
display(tuned_feature_importance.head(10))

# %% [markdown]
# ## 7. Model Saving and Versioning

# %% [markdown]
# ## 7. Model Saving and Versioning

# %%
def save_complete_model(model, preprocessor, file_path, metrics):
    """Save a model, preprocessor, and its metrics."""
    # For the ensemble object, the actual model is the meta_learner.
    # For a scikit-learn model, the model is the object itself.
    model_to_save = model.meta_learner if hasattr(model, 'meta_learner') else model

    model_system = {
        'model': model_to_save,
        'preprocessor': preprocessor,
        # Save the full model object if it's the ensemble, for full analysis
        'full_ensemble_object': model if hasattr(model, 'meta_learner') else None,
        'performance_metrics': metrics,
        'timestamp': '2025-11-12 13:27:59'
    }
    joblib.dump(model_system, file_path)
    print(f"✅ Complete model system saved to: {file_path}")

# --- SAVE BOTH THE BEST MODEL AND THE STACKED ENSEMBLE ---

# Create a single timestamp for this run based on the provided time
timestamp = '20251112_132759'

# 1. Save the BEST PERFORMING MODEL (Tuned RF)
best_model_path = f"/content/drive/MyDrive/disease_dataset/typhoid/typhoid_BEST_MODEL_v{timestamp}.pkl"
save_complete_model(best_rf_model, preprocessor, best_model_path, tuned_rf_results)

# 2. Save the STACKED ENSEMBLE MODEL
# We pass the entire 'ensemble' object, which contains the base models and meta-learner
stacked_model_path = f"/content/drive/MyDrive/disease_dataset/typhoid/typhoid_STACKED_ENSEMBLE_v{timestamp}.pkl"
save_complete_model(ensemble, preprocessor, stacked_model_path, ensemble_results)

# Also save the metrics comparison table to a readable CSV file
metrics_csv_path = f"/content/drive/MyDrive/disease_dataset/typhoid/typhoid_metrics_comparison_v{timestamp}.csv"
comparison_df.to_csv(metrics_csv_path, index=False)
print(f"✅ Performance metrics comparison saved to CSV: {metrics_csv_path}")

print(f"\n🎉 TYPHOID PREDICTION PROJECT WITH TUNING COMPLETED SUCCESSFULLY!")