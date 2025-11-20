import joblib
import logging
import os
import sys
import __main__  # Import the __main__ module

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from app.core.config import settings
from app.utils.data_preprocessors import (
    MalariaDataPreprocessor,
    DiabetesDataPreprocessor,
    TyphoidDataPreprocessor
)
# --- FIX: Import both custom classes ---
from app.utils.model_architectures import ClinicalStackedEnsemble, ComprehensiveEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.models = {}
        self.is_loaded = False

    def load_models(self):
        """
        Load all trained models, dynamically patching the __main__ namespace for all
        custom classes required for unpickling.
        """
        logger.info("Attempting to load machine learning models by patching __main__...")

        model_definitions = {
            "diabetes": {
                "path": os.path.join(settings.MODELS_DIR, "diabetes_model.pkl"),
                "preprocessor_class": DiabetesDataPreprocessor
            },
            "malaria": {
                "path": os.path.join(settings.MODELS_DIR, "malaria_model.pkl"),
                "preprocessor_class": MalariaDataPreprocessor
            },
            "typhoid": {
                "path": os.path.join(settings.MODELS_DIR, "typhoid_model.pkl"),
                "preprocessor_class": TyphoidDataPreprocessor
            },
        }

        try:
            for disease, definition in model_definitions.items():
                path = definition["path"]
                correct_preprocessor_class = definition["preprocessor_class"]

                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model file for '{disease}' not found at {path}")

                logger.info(f"Loading model for '{disease}'...")

                # --- THE DEFINITIVE FIX: Patch all three classes into __main__ ---
                setattr(__main__, 'DataPreprocessor', correct_preprocessor_class)
                setattr(__main__, 'ClinicalStackedEnsemble', ClinicalStackedEnsemble)
                setattr(__main__, 'ComprehensiveEvaluator', ComprehensiveEvaluator)

                # LOAD MODEL: joblib.load() will now find all necessary classes.
                self.models[disease] = joblib.load(path)
                logger.info(f" -> Successfully loaded '{disease}' model.")

                # CLEANUP: Remove the attributes from __main__.
                delattr(__main__, 'DataPreprocessor')
                delattr(__main__, 'ClinicalStackedEnsemble')
                delattr(__main__, 'ComprehensiveEvaluator')

            self.is_loaded = True
            logger.info("✅ All models loaded successfully.")

        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            self.is_loaded = False
        except Exception as e:
            logger.error(f"❌ An unexpected error occurred while loading models: {e}")
            logger.error("This can happen if a class definition is incorrect or missing.")
            self.is_loaded = False
            # Ensure cleanup happens even on error
            if hasattr(__main__, 'DataPreprocessor'): delattr(__main__, 'DataPreprocessor')
            if hasattr(__main__, 'ClinicalStackedEnsemble'): delattr(__main__, 'ClinicalStackedEnsemble')
            if hasattr(__main__, 'ComprehensiveEvaluator'): delattr(__main__, 'ComprehensiveEvaluator')

# Create a single, shared instance
model_loader = ModelLoader()