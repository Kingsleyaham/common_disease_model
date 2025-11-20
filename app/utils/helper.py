import pandas as pd

from app.constants.gender import Gender
from app.schemas.symptoms import PatientInfo


def preprocess_malaria_data(patient: PatientInfo) -> pd.DataFrame:
    """Preprocess data for malaria prediction"""
    features = {
        'Sex': 1 if patient.gender == Gender.MALE else 0,
        'Age': patient.age,
        'Hemoglobin(Hb%)': patient.lab_data.hemoglobin or 13.5,
        'Total WBC count(/cumm)': patient.lab_data.wbc_count or 7000,
        'Neutrophils': patient.lab_data.neutrophils or 55,
        'Lymphocytes': patient.lab_data.lymphocytes or 35,
        'Total Cir.Eosinophils': 200,
        'HTC/PCV(%)': patient.lab_data.hematocrit or 42.0,
        'MCH(pg)': 30.0,
        'MCHC(g/dl)': 33.0,
        'RDW-CV(%)': 13.5,
        'Platelet Count': patient.lab_data.platelets or 250000
    }

    df = pd.DataFrame([features])

    # Add engineered features
    df['Lymphocyte_Neutrophil_Ratio'] = df['Lymphocytes'] / (df['Neutrophils'] + 1e-8)
    df['Platelet_WBC_Ratio'] = df['Platelet Count'] / (df['Total WBC count(/cumm)'] + 1e-8)

    return df