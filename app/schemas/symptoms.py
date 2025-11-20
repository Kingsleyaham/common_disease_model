from pydantic import BaseModel, Field
from typing import Literal

# This defines the structure for a diabetes prediction request using snake_case
class DiabetesSymptoms(BaseModel):
    model_config = {"populate_by_name": True}

    age: int = Field(..., alias='Age')
    gender: Literal['Male', 'Female'] = Field(..., alias='Gender')
    polyuria: Literal['Yes', 'No'] = Field(..., alias='Polyuria')
    polydipsia: Literal['Yes', 'No'] = Field(..., alias='Polydipsia')
    sudden_weight_loss: Literal['Yes', 'No'] = Field(..., alias='sudden weight loss')
    weakness: Literal['Yes', 'No'] = Field(..., alias='weakness')
    polyphagia: Literal['Yes', 'No'] = Field(..., alias='Polyphagia')
    genital_thrush: Literal['Yes', 'No'] = Field(..., alias='Genital thrush')
    visual_blurring: Literal['Yes', 'No'] = Field(..., alias='visual blurring')
    itching: Literal['Yes', 'No'] = Field(..., alias='Itching')
    irritability: Literal['Yes', 'No'] = Field(..., alias='Irritability')
    delayed_healing: Literal['Yes', 'No'] = Field(..., alias='delayed healing')
    partial_paresis: Literal['Yes', 'No'] = Field(..., alias='partial paresis')
    muscle_stiffness: Literal['Yes', 'No'] = Field(..., alias='muscle stiffness')
    alopecia: Literal['Yes', 'No'] = Field(..., alias='Alopecia')
    obesity: Literal['Yes', 'No'] = Field(..., alias='Obesity')


# This defines the structure for a malaria prediction request using snake_case
class MalariaSymptoms(BaseModel):
    model_config = {"populate_by_name": True}

    age: float = Field(..., alias='Age')
    sex: Literal['Male', 'Female'] = Field(..., alias='Sex')
    hemoglobin_hb_percent: float = Field(..., alias='Hemoglobin(Hb%)')
    total_wbc_count: float = Field(..., alias='Total WBC count(/cumm)')
    neutrophils: float = Field(..., alias='Neutrophils')
    lymphocytes: float = Field(..., alias='Lymphocytes')
    total_cir_eosinophils: float = Field(..., alias='Total Cir.Eosinophils')
    htc_pcv_percent: float = Field(..., alias='HTC/PCV(%)')
    mch_pg: float = Field(..., alias='MCH(pg)')
    mchc_g_dl: float = Field(..., alias='MCHC(g/dl)')
    rdw_cv_percent: float = Field(..., alias='RDW-CV(%)')
    platelet_count: float = Field(..., alias='Platelet Count')


# This defines the structure for a typhoid prediction request using snake_case
class TyphoidSymptoms(BaseModel):
    model_config = {"populate_by_name": True}

    age: float = Field(..., alias='Age')
    gender: Literal['Male', 'Female'] = Field(..., alias='Gender')
    blood_group: str = Field(..., alias='Blood Group')
    esr: str = Field(..., alias='ESR')
    wbc_count: float = Field(..., alias='WBC Count')
    widal_test: str = Field(..., alias='Widal Test')
