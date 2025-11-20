import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder

# The base class is now simpler.
class BaseDataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.feature_names = None

# ... (The rest of the file is the same as the last version, just without the `load_preprocessor` method call in each class)
# The full content of the other classes remains as it was in the previous robust version.
class DiabetesDataPreprocessor(BaseDataPreprocessor):
    def transform_for_prediction(self, input_data):
        # ... (same logic as before)
        data = {col: [0] for col in self.feature_names}
        patient_df = pd.DataFrame(data)
        symptoms = input_data.dict(by_alias=True)
        for key, value in symptoms.items():
            if key in patient_df.columns:
                if key not in ['Age', 'Gender']: patient_df[key] = 1 if value == 'Yes' else 0
                elif key == 'Gender': patient_df[key] = 1 if value == 'Male' else 0
                else: patient_df[key] = value
        if 'Age' in patient_df.columns:
            patient_df[['Age']] = self.scaler.transform(patient_df[['Age']])
        return patient_df[self.feature_names]

class MalariaDataPreprocessor(BaseDataPreprocessor):
    def transform_for_prediction(self, input_data):
        # ... (same logic as before)
        patient_data = input_data.dict(by_alias=True)
        patient_df = pd.DataFrame([patient_data])
        patient_df['Sex'] = patient_df['Sex'].map({'Male': 1, 'Female': 0})
        patient_df = patient_df[self.feature_names]
        patient_df_scaled = self.scaler.transform(patient_df)
        return pd.DataFrame(patient_df_scaled, columns=self.feature_names)

class TyphoidDataPreprocessor(BaseDataPreprocessor):
    def _engineer_widal_features(self, df):
        # ... (same logic as before)
        def extract_dilution(text, antigen):
            if not isinstance(text, str): return 0
            match = re.search(f"{antigen}\\s*=\\s*1:(\\d+)", text, re.IGNORECASE)
            return float(match.group(1)) if match else 0
        df['Widal_TO'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'TO'))
        df['Widal_TH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'TH'))
        df['Widal_AH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'AH'))
        df['Widal_BH'] = df['Widal Test'].apply(lambda x: extract_dilution(x, 'BH'))
        return df

    def transform_for_prediction(self, input_data):
        # ... (same logic as before)
        final_df = pd.DataFrame(columns=self.feature_names, data=[[0]*len(self.feature_names)])
        symptoms = input_data.dict(by_alias=True)
        blood_group_col = f"Blood Group_{symptoms['Blood Group'].strip()}"
        if blood_group_col in final_df.columns: final_df[blood_group_col] = 1
        esr_col = f"ESR_{symptoms['ESR'].strip()}"
        if esr_col in final_df.columns: final_df[esr_col] = 1
        if 'Gender' in final_df.columns: final_df['Gender'] = 1 if symptoms['Gender'] == 'Male' else 0
        if 'Age' in final_df.columns: final_df['Age'] = symptoms['Age']
        if 'WBC Count' in final_df.columns: final_df['WBC Count'] = symptoms['WBC Count']
        widal_df = self._engineer_widal_features(pd.DataFrame([symptoms]))
        for col in ['Widal_TO', 'Widal_TH', 'Widal_AH', 'Widal_BH']:
            if col in final_df.columns: final_df[col] = widal_df[col]
        numerical_cols_to_scale = [col for col in ['Age', 'WBC Count', 'Widal_TO', 'Widal_TH', 'Widal_AH', 'Widal_BH'] if col in final_df.columns]
        if numerical_cols_to_scale:
            final_df[numerical_cols_to_scale] = self.scaler.transform(final_df[numerical_cols_to_scale])
        return final_df[self.feature_names]
