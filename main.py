from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import time
from fastapi.responses import StreamingResponse

rf_model_loaded = joblib.load('models/rf_model.pkl')

app = FastAPI()

class DataInput(BaseModel):
    edad_: float
    sexo_: int
    inten_prev: int
    prob_parej: int
    enfermedad_cronica: int
    prob_econo: int
    muerte_fam: int
    esco_educ: int
    prob_legal: int
    suici_fm_a: int
    maltr_fps: int
    prob_labor: int
    prob_famil: int
    prob_consu: int
    hist_famil: int
    idea_suici: int
    plan_suici: int
    antec_tran: int
    tran_depre: int
    trast_personalidad: int
    trast_bipolaridad: int
    esquizofre: int
    antec_v_a: int
    abuso_alco: int

@app.post("/predict")
def predict(data: DataInput):
    input_data = np.array([[
        data.edad_, data.sexo_, data.inten_prev, data.prob_parej, data.enfermedad_cronica,
        data.prob_econo, data.muerte_fam, data.esco_educ, data.prob_legal, data.suici_fm_a,
        data.maltr_fps, data.prob_labor, data.prob_famil, data.prob_consu, data.hist_famil,
        data.idea_suici, data.plan_suici, data.antec_tran, data.tran_depre, data.trast_personalidad,
        data.trast_bipolaridad, data.esquizofre, data.antec_v_a, data.abuso_alco
    ]])

    prediction = rf_model_loaded.predict(input_data)

    return {"prediction": prediction[0]}

@app.post("/predict/streaming", response_class=StreamingResponse)
async def streaming(data: DataInput):
    input_data = np.array([[
        data.edad_,data.sexo_, data.inten_prev, data.prob_parej, data.enfermedad_cronica, data.prob_econo,
        data.muerte_fam, data.esco_educ, data.prob_legal, data.suici_fm_a, data.maltr_fps, data.prob_labor,
        data.prob_famil, data.prob_consu, data.hist_famil, data.idea_suici, data.plan_suici, data.antec_tran,
        data.tran_depre, data.trast_personalidad, data.trast_bipolaridad, data.esquizofre, data.antec_v_a, data.abuso_alco
    ]])

    prediction = rf_model_loaded.predict(input_data)

    condition = "Vivo" if prediction[0] == 1 else "Muerto"
    factors = [
        "edad", "sexo", "intentos previos de suicidio", "conflictos con pareja o expareja", "enfermedad crónica",
        "problemas económicos", "muerte de un familiar", "educación", "problemas legales", "suicidio de un amigo o familiar",
        "maltrato físico/psicológico/sexual", "problemas laborales", "problemas familiares", "consumo de sustancias psicoactivas",
        "antecedentes familiares", "ideación suicida", "plan suicida", "antecedente de trastorno psiquiátrico",
        "trastorno depresivo", "trastorno de personalidad", "trastorno bipolar", "esquizofrenia", "violencia o abuso", "abuso de alcohol"
    ]
    factor_values = [
        "%.1f años" % data.edad_,
        "Masculino" if data.sexo_ == 1 else "Femenino",
        "Sí" if data.inten_prev == 1 else "No",
        "Sí" if data.prob_parej == 1 else "No",
        "Sí" if data.enfermedad_cronica == 1 else "No",
        "Sí" if data.prob_econo == 1 else "No",
        "Sí" if data.muerte_fam == 1 else "No",
        "Sí" if data.esco_educ == 1 else "No",
        "Sí" if data.prob_legal == 1 else "No",
        "Sí" if data.suici_fm_a == 1 else "No",
        "Sí" if data.maltr_fps == 1 else "No",
        "Sí" if data.prob_labor == 1 else "No",
        "Sí" if data.prob_famil == 1 else "No",
        "Sí" if data.prob_consu == 1 else "No",
        "Sí" if data.hist_famil == 1 else "No",
        "Sí" if data.idea_suici == 1 else "No",
        "Sí" if data.plan_suici == 1 else "No",
        "Sí" if data.antec_tran == 1 else "No",
        "Sí" if data.tran_depre == 1 else "No",
        "Sí" if data.trast_personalidad == 1 else "No",
        "Sí" if data.trast_bipolaridad == 1 else "No",
        "Sí" if data.esquizofre == 1 else "No",
        "Sí" if data.antec_v_a == 1 else "No",
        "Sí" if data.abuso_alco == 1 else "No"
    ]
    
    def generate_streaming_response():
        yield f"Teniendo en cuenta los siguientes factores:\n"
        for factor, value in zip(factors, factor_values):
            time.sleep(0.5)
            yield f"- {factor}: {value}\n"
        
        time.sleep(0.5)
        yield f"\nSe tiene una probabilidad de tener una condición final: {condition}.\n"

    return StreamingResponse(generate_streaming_response(), media_type="text/plain")
