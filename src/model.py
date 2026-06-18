import torch
import shap
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, pipeline

from src.config import MODEL_PATH, BASE_MODEL

@st.cache_resource(show_spinner="🔄 Memuat model AI ke memori (Mohon tunggu)...")
def load_model():
    tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)
    model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
    device    = 0 if torch.cuda.is_available() else -1

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        top_k=None,
    )
    explainer = shap.Explainer(pipe)
    return pipe, explainer