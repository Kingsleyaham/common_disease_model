# %% [markdown]
# # Malaria Prediction - Complete Implementation
# ## Using Clinical Laboratory Data
# 
# **Research Project:** Predictive Diagnostic Model for Malaria using Stacked Ensemble Machine Learning

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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, 
                           roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

print("✅ All packages imported successfully!")

# %% [markdown]
# ## 2. Data Loading from Google Drive

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# %%
# Load your actual malaria dataset
def load_malaria_data():
    """Load your malaria laboratory dataset from Google Drive"""
    try:
        # IMPORTANT: Update this path to your actual malaria file location
        file_path = "/content/drive/MyDrive/disease_dataset/malaria_data.csv" #<-- MAKE SURE THIS FILENAME IS CORRECT
        df = pd.read_csv(file_path)
        print(f"✅ Dataset loaded successfully: {len(df)} records")
        
        # Display basic info
        print(f"📊 Dataset Shape: {df.shape}")
        print(f"🎯 Target distribution:")
        print(df['Result'].value_counts()) #<-- Changed to 'Result' column
        
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        print("Please check the file path and try again.")
        return None

# Load the data
df = load_malaria_data()

if df is not None:
    display(df.head())
    print("\n📋 Column names:")
    print(df.columns.tolist())

# %% [markdown]
# ## 3. Data Preprocessing and Augmentation

# %%
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def preprocess_data(self, df, target_col='Result', test_size=0.2, augment_data=True):
        """Preprocess and optionally augment the malaria data"""
        print("🔄 Preprocessing data...")
        
        # Create a copy
        df_processed = df.copy()

        # --- START: FIX FOR THE ERROR ---
        # 1. Standardize the target column to prevent multiple classes
        print(f"   Original unique values in '{target_col}': {df_processed[target_col].unique()}")
        df_processed[target_col] = df_processed[target_col].str.strip().str.lower()
        print(f"   Standardized unique values in '{target_col}': {df_processed[target_col].unique()}")
        
        # 2. Explicitly map to 1 (positive) and 0 (negative)
        df_processed[target_col] = df_processed[target_col].map({'positive': 1, 'negative': 0})

        # 3. Drop any rows that couldn't be mapped (if any rogue values existed)
        original_len = len(df_processed)
        df_processed.dropna(subset=[target_col], inplace=True)
        if len(df_processed) < original_len:
            print(f"   Dropped {original_len - len(df_processed)} rows with unmappable target values.")
        df_processed[target_col] = df_processed[target_col].astype(int)
        # --- END: FIX FOR THE ERROR ---

        # Handle potential missing values in features with median imputation
        for col in df_processed.select_dtypes(include=np.number).columns:
            if df_processed[col].isnull().sum() > 0:
                print(f"   Imputing missing values in '{col}' with median.")
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
        
        # Encode categorical 'Sex' column
        if 'Sex' in df_processed.columns:
            df_processed['Sex'] = LabelEncoder().fit_transform(df_processed['Sex'])
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        self.feature_names = X.columns.tolist()
        
        print(f"Original class distribution: {y.value_counts().to_dict()}")
        
        # Data augmentation with SMOTE (if enabled)
        if augment_data:
            X, y = self._augment_data(X, y)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale all numerical features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame to keep column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)

        print(f"✅ Final dataset: {len(X_train_scaled)} training, {len(X_test_scaled)} test samples")
        print(f"📊 Training class distribution: {y_train.value_counts().to_dict()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _augment_data(self, X, y):
        """Augment data using SMOTE to handle class imbalance"""
        print("🔄 Augmenting data with SMOTE...")
        
        original_counts = y.value_counts()
        # k_neighbors must be less than the number of samples in the smallest class
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
# ## 4. Stacked Ensemble Implementation
# NOTE: This section is identical to the diabetes script, as the methodology is the same.

# %%
class ClinicalStackedEnsemble:
    def __init__(self):
        self.base_models = []
        self.meta_learner = None
        self.cv_scores = {}
        self.feature_importance = None
        
    def initialize_base_models(self):
        """Initialize diverse base learners optimized for clinical data"""
        self.base_models = [
            ('random_forest', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)),
            ('xgboost', XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='logloss')),
            ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=7, weights='distance', n_jobs=-1)),
            ('logistic', LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))
        ]
        print("✅ Base models initialized:", [name for name, _ in self.base_models])
    
    def train_base_models(self, X_train, y_train):
        """Train all base models with cross-validation"""
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
        """Generate meta-features using cross-validation predictions"""
        print("\n🔄 Generating meta-features...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_features = np.hstack([
            cross_val_predict(model, X, y, cv=skf, method='predict_proba')
            for name, model in self.base_models
        ])
        print(f"✅ Meta-features shape: {meta_features.shape}")
        return meta_features
    
    def train_meta_learner(self, meta_features, y):
        """Train the meta-learner on base model predictions"""
        print("\n🔧 Training meta-learner...")
        self.meta_learner = LogisticRegression(random_state=42, max_iter=1000)
        self.meta_learner.fit(meta_features, y)
        print(f"✅ Meta-learner training completed. Score: {self.meta_learner.score(meta_features, y):.4f}")
    
    def train_stacked_ensemble(self, X_train, y_train, feature_names): # <-- ADD feature_names here
        """Train the complete stacked ensemble"""
        print("=" * 60, "\n🚀 TRAINING STACKED ENSEMBLE MODEL\n" + "=" * 60)
        start_time = time.time()
        
        # Store feature names for later use
        self.feature_names = feature_names # <-- STORE the names
        
        self.initialize_base_models()
        self.train_base_models(X_train, y_train)
        meta_features = self.generate_meta_features(X_train, y_train)
        self.train_meta_learner(meta_features, y_train)
        self._calculate_feature_importance() # This will now work
        print(f"\n✅ Stacked ensemble training completed! Total time: {time.time() - start_time:.2f}s")
    
    def predict(self, X):
        """Make predictions using stacked ensemble"""
        base_predictions = np.hstack([model.predict_proba(X) for name, model in self.base_models])
        return self.meta_learner.predict(base_predictions)
    
    def predict_proba(self, X):
        """Get probability predictions"""
        base_predictions = np.hstack([model.predict_proba(X) for name, model in self.base_models])
        return self.meta_learner.predict_proba(base_predictions)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance from Random Forest"""
        rf_model = next((model for name, model in self.base_models if name == 'random_forest'), None)
        if rf_model:
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

# %%
# Train the stacked ensemble
print("🎯 Starting Stacked Ensemble Training for Malaria...")
ensemble = ClinicalStackedEnsemble()
ensemble.train_stacked_ensemble(X_train, y_train, preprocessor.feature_names)

# %% [markdown]
# ## 5. Comprehensive Model Evaluation
# NOTE: This section is also identical, as the evaluation metrics are universal.

# %%
class ComprehensiveEvaluator:
    def __init__(self):
        self.results = {}
        self.metrics_history = []
        
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Comprehensive evaluation with all metrics"""
        print(f"\n{'='*50}\n📊 EVALUATING: {model_name.upper()}\n{'='*50}")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba[:, 1])
        }
        self.results[model_name] = results
        self.metrics_history.append(results)
        
        self._print_detailed_results(results)
        self._plot_comprehensive_evaluation(y_test, y_pred, y_proba, model_name, results['roc_auc'])
        return results
    
    def _print_detailed_results(self, results):
        print(f"🎯 PERFORMANCE METRICS:\n"
              f"   Accuracy:    {results['accuracy']:.4f}\n"
              f"   Precision:   {results['precision']:.4f}\n"
              f"   Recall:      {results['recall']:.4f}\n"
              f"   F1-Score:    {results['f1_score']:.4f}\n"
              f"   ROC AUC:     {results['roc_auc']:.4f}")
    
    def _plot_comprehensive_evaluation(self, y_test, y_pred, y_proba, model_name, roc_auc):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Comprehensive Evaluation: {model_name}', fontsize=16)
        
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', ax=axes[0], cmap='Blues')
        axes[0].set_title('Normalized Confusion Matrix')
        
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic (ROC) Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def compare_base_models(self, ensemble, X_test, y_test):
        print("\n" + "="*60 + "\n🔍 COMPARING BASE MODEL PERFORMANCE\n" + "="*60)
        return {name: self.evaluate_model(model, X_test, y_test, f"Base_{name}") for name, model in ensemble.base_models}

    def create_comparison_table(self):
        return pd.DataFrame(self.metrics_history).round(4)

# %%
# Run evaluations
evaluator = ComprehensiveEvaluator()
ensemble_results = evaluator.evaluate_model(ensemble, X_test, y_test, "Stacked_Ensemble")
base_results = evaluator.compare_base_models(ensemble, X_test, y_test)

# %% [markdown]
# ## 6. Performance Comparison and Final Analysis

# %%
print("\n" + "="*70 + "\n🏆 FINAL PERFORMANCE COMPARISON\n" + "="*70)
comparison_df = evaluator.create_comparison_table().sort_values('f1_score', ascending=False)
print("\n📈 Model Performance Ranking (Sorted by F1-Score):")
display(comparison_df)

if ensemble.feature_importance is not None:
    print(f"\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
    display(ensemble.feature_importance.head(10))

# %% [markdown]
# ## 7. Model Saving and Versioning

# %%
def save_complete_model(ensemble, preprocessor, evaluator, file_path):
    """Save the complete model system for deployment"""
    model_system = {
        'ensemble': ensemble, 'preprocessor': preprocessor, 'evaluator': evaluator,
        'feature_names': preprocessor.feature_names, 'performance_metrics': evaluator.results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    joblib.dump(model_system, file_path)
    print(f"✅ Complete model system saved to: {file_path}")

# Create a timestamp for unique versioning
timestamp = time.strftime("%Y%m%d_%H%M%S")
model_path = f"/content/drive/MyDrive/disease_dataset/malaria/malaria_ensemble_v{timestamp}.pkl"
save_complete_model(ensemble, preprocessor, evaluator, model_path)

# Also save metrics to a readable CSV file
metrics_csv_path = f"/content/drive/MyDrive/disease_dataset/malaria/malaria_metrics_v{timestamp}.csv"
comparison_df.to_csv(metrics_csv_path, index=False)
print(f"✅ Performance metrics saved to CSV: {metrics_csv_path}")

print(f"\n🎉 MALARIA PREDICTION PROJECT COMPLETED SUCCESSFULLY!")
