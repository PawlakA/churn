import os
import pandas as pd
import mlflow.sklearn

MODEL_DIR = "models"

import glob
# local_model_paths = glob.glob("./*/artifacts/model")
root_dir = "."
local_model_paths = glob.glob(os.path.join(root_dir, '**', 'artifacts'), recursive=True)
if local_model_paths:
    latest_model = max(local_model_paths, key=os.path.getmtime)
    model = mlflow.sklearn.load_model(latest_model)
    MODEL_DIR = latest_model
    print(f"✅ Fallback: Loaded model from {latest_model}")
else:
    raise Exception("No model found in local mlruns")

def predict(input_dict: dict) -> int:

    df = pd.DataFrame([input_dict])

    probs = model.predict_proba(df)

    THRESHOLD = 0.35
    preds = (probs >= THRESHOLD).astype(int)
    result = preds.tolist()[0][1]

    return result