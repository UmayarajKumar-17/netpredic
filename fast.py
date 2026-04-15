from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load(r"E:\netcore-copilot\4netcore\inventory_model.pkl")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])

    predicted_demand = model.predict(df)[0]

    lead_time = data["lead_time"]

    std_dev = 10  

    safety_stock = 1.65 * std_dev * (lead_time ** 0.5)

    reorder_point = (predicted_demand * lead_time) + safety_stock


    current_stock = data["stock"]

    if current_stock <= reorder_point:
        recommendation = "Reorder now"
    else:
        recommendation = "Stock is sufficient"

    return {
        "predicted_demand": round(predicted_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommendation": recommendation
    }