import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

file_path = 'sivigila_intsuicidio.csv'

columns_to_transform = [
    'con_fin_', 'inten_prev', 'prob_parej', 'enfermedad_cronica', 'prob_econo', 'muerte_fam',
    'esco_educ', 'prob_legal', 'suici_fm_a', 'maltr_fps', 'prob_labor', 'prob_famil', 'prob_consu', 'hist_famil',
    'idea_suici', 'plan_suici', 'antec_tran', 'tran_depre', 'trast_personalidad', 'trast_bipolaridad', 'esquizofre',
    'antec_v_a', 'abuso_alco'
]

selected_columns = [
    'con_fin_', 'edad_','sexo_', 'inten_prev', 'prob_parej', 'enfermedad_cronica', 'prob_econo', 'muerte_fam',
    'esco_educ', 'prob_legal', 'suici_fm_a', 'maltr_fps', 'prob_labor', 'prob_famil', 'prob_consu', 'hist_famil',
    'idea_suici', 'plan_suici', 'antec_tran', 'tran_depre', 'trast_personalidad', 'trast_bipolaridad', 'esquizofre',
    'antec_v_a', 'abuso_alco'
]

class Transformation:
    def __init__(self):
        self.df = pd.read_csv(file_path, low_memory=False)

    def transform_column(self, col, is_sex_column=False):
        if is_sex_column:
            return col.apply(lambda x: 1 if x == 'M' else (0 if x == 'F' else None))
        else:
            return col.apply(lambda x: 1 if x == 1 else (0 if x == 2 else None))
        
    def get_dataset(self):
        for col in columns_to_transform:
            self.df[col] = self.transform_column(self.df[col])

        self.df['sexo_'] = self.transform_column(self.df['sexo_'], is_sex_column=True)

        imputer = SimpleImputer(strategy='most_frequent')

        self.df[selected_columns] = imputer.fit_transform(self.df[selected_columns])

        data_selected = self.df[selected_columns].copy()

        minority_class = data_selected[data_selected['con_fin_'] == 0.0]

        num_synthetic_samples = data_selected['con_fin_'].value_counts()[1]  # Queremos que el número de muestras sintéticas sea igual al de 'vivos' (1.0)

        synthetic_data = minority_class.sample(num_synthetic_samples, replace=True, random_state=42)

        for col in synthetic_data.columns:
            if col != 'con_fin_' and synthetic_data[col].dtype in [np.float64, np.int64]:  
                noise = np.random.normal(0, 0.1, size=synthetic_data[col].shape)  
                synthetic_data[col] = synthetic_data[col] + noise

        categorical_columns = ['sexo_', 'inten_prev', 'prob_parej', 'enfermedad_cronica', 'prob_econo', 'muerte_fam', 
                              'esco_educ', 'prob_legal', 'suici_fm_a', 'maltr_fps', 'prob_labor', 'prob_famil', 'prob_consu', 
                              'hist_famil', 'idea_suici', 'plan_suici', 'antec_tran', 'tran_depre', 'trast_personalidad', 
                              'trast_bipolaridad', 'esquizofre', 'antec_v_a', 'abuso_alco']

        for col in categorical_columns:
            synthetic_data[col] = synthetic_data[col].round().astype(int)

        synthetic_data['edad_'] = synthetic_data['edad_'].apply(lambda x: max(13, min(round(x), 60))) 

        df_balanced = pd.concat([data_selected, synthetic_data], axis=0)

        return df_balanced
