from enum import Enum

class DiseaseType(str, Enum):
    MALARIA = "malaria"
    DIABETES = "diabetes"
    TYPHOID = "typhoid"
    ALL = "all"