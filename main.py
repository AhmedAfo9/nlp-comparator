import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import re
import os

app = FastAPI(title="Sovereign Linguistic Comparator Engine", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class AnalysisRequest(BaseModel):
    ai_text: str
    human_text: str

AI_BUZZWORDS = [
    "delve", "crucial", "testament", "tapestry", "paramount", "furthermore",
    "moreover", "vital", "intricate", "multifaceted", "underscores", "pivotal",
    "foster", "interplay", "beacon", "in conclusion", "it is important to note",
    "plays a vital role", "shed light", "comprehensive"
]

def count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r'[aeiouy]{1,2}', word))
    if word.endswith('e') and not word.endswith('le') and len(word) > 2:
        count -= 1
    return max(1, count)

def compute_deep_nlp_metrics(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return {}
    
    words = [t.text.lower() for t in tokens]
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
    
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    min_sent_len = int(np.min(sent_lengths)) if sent_lengths else 0
    max_sent_len = int(np.max(sent_lengths)) if sent_lengths else 0
    std_dev_sent = float(np.std(sent_lengths)) if sent_lengths else 0.0
    
    total_syllables = sum(count_syllables(w) for w in words)
    asl = avg_sent_len
    asw = total_syllables / len(words) if words else 0
    flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    stop_words = [t for t in tokens if t.is_stop]
    stop_word_ratio = len(stop_words) / len(tokens)
    
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0, "CONJ": 0}
    for t in tokens:
        pos = t.pos_
        if pos in ["CCONJ", "SCONJ"]:
            pos_counts["CONJ"] += 1
        elif pos in pos_counts:
            pos_counts[pos] += 1
            
    pos_ratios = {k: round((v / len(tokens)) * 100, 2) for k, v in pos_counts.items()}
    
    found_buzzwords = [w for w in AI_BUZZWORDS if re.search(r'\b' + re.escape(w) + r'\b', text.lower())]
    
    puncts = [t.text for t in doc if t.is_punct]
    punct_ratio = round((len(puncts) / len(doc)) * 100, 2) if len(doc) > 0 else 0

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "lexical_diversity_ttr": round(ttr * 100, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "burstiness_variance": round(sent_variance, 2),
        "min_sentence_length": min_sent_len,
        "max_sentence_length": max_sent_len,
        "sentence_std_dev": round(std_dev_sent, 2),
        "flesch_readability": round(flesch_score, 2),
        "stopword_percentage": round(stop_word_ratio * 100, 2),
        "pos_distribution": pos_ratios,
        "ai_buzzwords_detected": found_buzzwords,
        "punctuation_percentage": punct_ratio
    }

def generate_english_academic_verdict(ai_m, hum_m):
    reasons = []
    
    # Burstiness Variance Diagnosis
    if hum_m["burstiness_variance"] > ai_m["burstiness_variance"]:
        reasons.append(f"Rhythmic Burstiness: Human reference exhibits higher sentence variance ({hum_m['burstiness_variance']} vs {ai_m['burstiness_variance']}), confirming natural, dynamic clause length distribution.")
    else:
        reasons.append(f"Rhythmic Monotony: AI sample demonstrates artificial sentence length uniformity ({ai_m['burstiness_variance']} vs {hum_m['burstiness_variance']}), characteristic of LLM token probability modeling.")
        
    # POS Lexical Density Diagnosis
    if ai_m["pos_distribution"]["ADJ"] > hum_m["pos_distribution"]["ADJ"]:
        reasons.append(f"Adjectival Embellishment: AI sample relies on a higher descriptive adjective ratio ({ai_m['pos_distribution']['ADJ']}% vs {hum_m['pos_distribution']['ADJ']}%), reflecting standard machine stylistic inflation.")
        
    if hum_m["pos_distribution"]["VERB"] > ai_m["pos_distribution"]["VERB"]:
        reasons.append(f"Syntactic Agency: Human reference relies more on active verbs ({hum_m['pos_distribution']['VERB']}% vs {ai_m['pos_distribution']['VERB']}%), providing stronger prose dynamism.")
        
    # Detected AI Buzzwords
    if len(ai_m["ai_buzzwords_detected"]) > 0:
        reasons.append(f"Probabilistic Collocations: Detected {len(ai_m['ai_buzzwords_detected'])} high-probability AI marker word(s): [{', '.join(ai_m['ai_buzzwords_detected'])}].")

    return reasons

@app.post("/compare")
def compare_texts(req: AnalysisRequest):
    if not req.ai_text.strip() or not req.human_text.strip():
        raise HTTPException(status_code=400, detail="Both texts must be provided.")
    
    ai_results = compute_deep_nlp_metrics(req.ai_text)
    human_results = compute_deep_nlp_metrics(req.human_text)
    verdict_reasons = generate_english_academic_verdict(ai_results, human_results)
    
    return {
        "status": "success",
        "ai_metrics": ai_results,
        "human_metrics": human_results,
        "diagnostic_verdict": verdict_reasons
    }

@app.get("/")
def root():
    return {"status": "active", "message": "English Academic NLP Comparator API v3.0 Active"}
