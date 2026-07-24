import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="NLP Text Analyzer API", version="1.0")

# تفعيل CORS لتسمح لموقعك على Cloudflare بالاتصال بالباكإند
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل نموذج spaCy للغة الإنجليزية
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class AnalysisRequest(BaseModel):
    ai_text: str
    human_text: str

def compute_nlp_metrics(text: str):
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_punct and not token.is_space]
    if not tokens:
        return {}
    
    words = [t.text.lower() for t in tokens]
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0  # Type-Token Ratio
    
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_len_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0  # Burstiness
    
    stop_words = [t for t in tokens if t.is_stop]
    stop_word_ratio = len(stop_words) / len(tokens) if tokens else 0
    
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for t in tokens:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
            
    pos_ratios = {k: (v / len(tokens)) * 100 for k, v in pos_counts.items()} if tokens else {}
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "lexical_diversity_ttr": round(ttr * 100, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "burstiness_variance": round(sent_len_variance, 2),
        "stopword_percentage": round(stop_word_ratio * 100, 2),
        "pos_distribution": {k: round(v, 2) for k, v in pos_ratios.items()}
    }

@app.post("/compare")
def compare_texts(req: AnalysisRequest):
    if not req.ai_text.strip() or not req.human_text.strip():
        raise HTTPException(status_code=400, detail="Both AI and Human texts must be provided.")
    
    ai_results = compute_nlp_metrics(req.ai_text)
    human_results = compute_nlp_metrics(req.human_text)
    
    return {
        "status": "success",
        "ai_metrics": ai_results,
        "human_metrics": human_results
    }

@app.get("/")
def root():
    return {"status": "active", "message": "NLP Compare Engine is Live!"}
