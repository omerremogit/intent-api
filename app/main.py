from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
from datasets import load_dataset
import torch

app = FastAPI()

class Query(BaseModel):
    text: str

model_name = "omerremohug/intent-detection-distilbert"

# Load model + tokenizer from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the intent labels
clinc = load_dataset("clinc_oos", "plus")
label_list = clinc["train"].features["intent"].names

@app.post("/predict")
async def predict(query: Query):
    inputs = tokenizer(query.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        pred_label = label_list[pred_index]
        confidence = float(torch.max(probs))
    
    return {
        "intent_label": pred_label,
        "confidence": confidence
    }
