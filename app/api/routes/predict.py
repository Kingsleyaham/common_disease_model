from fastapi import APIRouter, HTTPException, status
from app.utils.model_loader import model_loader
from app.schemas.symptoms import DiabetesSymptoms, MalariaSymptoms, TyphoidSymptoms
# The preprocessor classes are still needed for type hinting and structure, but we won't instantiate them here.
from app.utils.data_preprocessors import DiabetesDataPreprocessor, MalariaDataPreprocessor, TyphoidDataPreprocessor

router = APIRouter()


def get_model_and_processor(disease: str):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded.")
    try:
        return model_loader.models[disease]
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model for '{disease}' not found.")


@router.post("/diabetes", summary="Predict Diabetes")
async def predict_diabetes(symptoms: DiabetesSymptoms):
    try:
        model_system = get_model_and_processor("diabetes")

        # FIX: Correctly get the model from 'ensemble' or 'model' key
        model = model_system.get('ensemble') or model_system.get('model')
        if model is None:
            raise ValueError("Could not find a valid model in the saved object.")

        # --- DEFINITIVE FIX ---
        # Use the preprocessor object directly as loaded by joblib. Do NOT create a new instance.
        processor = model_system['preprocessor']
        processed_data = processor.transform_for_prediction(symptoms)

        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = int(model.predict(processed_data)[0])

        return {"success": True, "disease": "diabetes", "prediction": "Positive" if prediction == 1 else "Negative",
                "confidence_score": float(prediction_proba[1])}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred during prediction: {str(e)}")


@router.post("/malaria", summary="Predict Malaria")
async def predict_malaria(symptoms: MalariaSymptoms):
    try:
        model_system = get_model_and_processor("malaria")
        model = model_system['model']

        # --- DEFINITIVE FIX ---
        # Use the preprocessor object directly.
        processor = model_system['preprocessor']
        processed_data = processor.transform_for_prediction(symptoms)

        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = int(model.predict(processed_data)[0])

        return {"success": True, "disease": "malaria", "prediction": "Positive" if prediction == 1 else "Negative",
                "confidence_score": float(prediction_proba[1])}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred during prediction: {str(e)}")


@router.post("/typhoid", summary="Predict Typhoid")
async def predict_typhoid(symptoms: TyphoidSymptoms):
    try:
        model_system = get_model_and_processor("typhoid")

        # Some of your training scripts used 'model', others 'ensemble'. This handles both.
        model = model_system.get('model') or model_system.get('ensemble')
        if model is None:
            raise ValueError("Could not find a valid model in the saved object.")

        # --- DEFINITIVE FIX ---
        # Use the preprocessor object directly. It already contains the fitted scaler.
        processor = model_system['preprocessor']
        processed_data = processor.transform_for_prediction(symptoms)

        prediction_proba = model.predict_proba(processed_data)[0]
        prediction = int(model.predict(processed_data)[0])

        return {"success": True, "disease": "typhoid", "prediction": "Positive" if prediction == 1 else "Negative",
                "confidence_score": float(prediction_proba[1])}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"An error occurred during prediction: {str(e)}")
