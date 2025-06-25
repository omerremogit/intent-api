import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Debug logs for Railway startup
print("🔍 Current Working Directory:", os.getcwd())
print("📁 Files in /app:", os.listdir("app") if os.path.exists("app") else "Missing")
print("📁 Files in /app/model:", os.listdir("app/model") if os.path.exists("app/model") else "Missing")

# Load model and tokenizer from local path
model_path = "app/model"

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/predict")
def predict(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = round(probs[0][pred_class].item(), 4)
    return {"predicted_intent": pred_class, "confidence": confidence}
