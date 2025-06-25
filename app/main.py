from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch

app = FastAPI()

class Query(BaseModel):
    text: str

# ✅ Replace this with your actual username if it's not correct
model_name = "omerremohug/intent-detection-distilbert"

# Load model + tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.post("/predict")
async def predict(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
    return {"intent_class": prediction, "confidence": float(torch.max(probs))}
