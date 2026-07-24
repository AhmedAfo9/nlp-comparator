import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import re
import os

app = FastAPI(title="Academic Stylometric Engine", version="4.0")

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
    "plays a vital role", "shed light", "comprehensive", "aligns with"
]

def count_syllables(word: str) -> int:
    word = word.lower()
    count = len(re.findall(r'[aeiouy]{1,2}', word))
    if word.endswith('e') and not word.endswith('le') and len(word) > 2:
        count -= 1
    return max(1, count)

def extract_detailed_linguistics(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return {}
    
    words = [t.text.lower() for t in tokens]
    unique_words = set(words)
    ttr = (len(unique_words) / len(words)) * 100 if words else 0
    
    # Sentences structure
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
    
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    min_sent_len = int(np.min(sent_lengths)) if sent_lengths else 0
    max_sent_len = int(np.max(sent_lengths)) if sent_lengths else 0
    std_dev_sent = float(np.std(sent_lengths)) if sent_lengths else 0.0
    
    # Readability
    total_syllables = sum(count_syllables(w) for w in words)
    asl = avg_sent_len
    asw = total_syllables / len(words) if words else 0
    flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    # Nominalization count (words ending with tion, ment, ance, ence, ity)
    nominalizations = [w for w in words if re.search(r'(tion|ment|ance|ence|ity|ness)$', w) and len(w) > 5]
    nom_ratio = round((len(nominalizations) / len(words)) * 100, 2) if words else 0
    
    # Passive Voice detection
    passive_verbs = [t.text for t in doc if t.dep_ in ["auxpass", "agent"]]
    
    # POS distribution & token lists
    pos_counts = {"NOUN": [], "VERB": [], "ADJ": [], "ADV": [], "PRON": [], "CONJ": []}
    for t in tokens:
        pos = t.pos_
        if pos in ["CCONJ", "SCONJ"]:
            pos_counts["CONJ"].append(t.text.lower())
        elif pos in pos_counts:
            pos_counts[pos].append(t.text.lower())
            
    pos_ratios = {k: round((len(v) / len(tokens)) * 100, 2) for k, v in pos_counts.items()}
    
    found_buzzwords = [w for w in AI_BUZZWORDS if re.search(r'\b' + re.escape(w) + r'\b', text.lower())]
    
    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "lexical_diversity_ttr": round(ttr, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "burstiness_variance": round(sent_variance, 2),
        "min_sentence_length": min_sent_len,
        "max_sentence_length": max_sent_len,
        "sentence_std_dev": round(std_dev_sent, 2),
        "flesch_readability": round(flesch_score, 2),
        "nominalization_ratio": nom_ratio,
        "nominalized_words_sample": list(set(nominalizations))[:8],
        "passive_constructions_count": len(passive_verbs),
        "pos_distribution": pos_ratios,
        "pos_samples": {
            "adjectives": list(set(pos_counts["ADJ"]))[:8],
            "verbs": list(set(pos_counts["VERB"]))[:8],
            "nouns": list(set(pos_counts["NOUN"]))[:8]
        },
        "ai_buzzwords_detected": found_buzzwords
    }

def generate_academic_deep_critique(ai, hum):
    reasons = []
    
    # 1. Burstiness Analysis
    if hum["burstiness_variance"] > ai["burstiness_variance"]:
        reasons.append(f"Rhythmic Burstiness & Cadence: Human text demonstrates significantly higher sentence length variance ({hum['burstiness_variance']} vs {ai['burstiness_variance']}). Human writing exhibits organic cadence alternating between brief assertions ({hum['min_sentence_length']} words) and complex syntactical structures ({hum['max_sentence_length']} words). Conversely, the AI text demonstrates robotic structural smoothing.")
    else:
        reasons.append(f"Rhythmic Monotony: AI sample shows constrained variance ({ai['burstiness_variance']}), characteristic of probabilistic Large Language Model decoding.")
        
    # 2. Nominalization & Style
    if ai["nominalization_ratio"] > hum["nominalization_ratio"]:
        reasons.append(f"Lexical Nominalization Index: AI text exhibits higher heavy nominalization ({ai['nominalization_ratio']}% vs {hum['nominalization_ratio']}%). Words such as [{', '.join(ai['nominalized_words_sample'][:4])}] convert active human actions into abstract noun phrases.")
    else:
        reasons.append(f"Action-Oriented Syntax: Human text maintains direct verbal momentum with lower nominalization ratio ({hum['nominalization_ratio']}%).")
        
    # 3. Adjectival & Verb Density
    reasons.append(f"Part-of-Speech Distribution: AI text contains {ai['pos_distribution']['ADJ']}% adjectives vs {hum['pos_distribution']['ADJ']}% in human text. AI relies on qualitative modifiers [{', '.join(ai['pos_samples']['adjectives'][:4])}], whereas human prose utilizes verbal agency ({hum['pos_distribution']['VERB']}% verbs).")
    
    # 4. Marker Collocations
    if len(ai["ai_buzzwords_detected"]) > 0:
        reasons.append(f"Probabilistic Marker Collocations: Detected {len(ai['ai_buzzwords_detected'])} classic LLM transition buzzwords: [{', '.join(ai['ai_buzzwords_detected'])}].")

    return reasons

@app.post("/compare")
def compare_texts(req: AnalysisRequest):
    if not req.ai_text.strip() or not req.human_text.strip():
        raise HTTPException(status_code=400, detail="Both texts are required.")
    
    ai_metrics = extract_detailed_linguistics(req.ai_text)
    human_metrics = extract_detailed_linguistics(req.human_text)
    deep_critique = generate_academic_deep_critique(ai_metrics, human_metrics)
    
    return {
        "status": "success",
        "ai_metrics": ai_metrics,
        "human_metrics": human_metrics,
        "diagnostic_verdict": deep_critique
    }

@app.get("/")
def root():
    return {"status": "active", "message": "Academic Stylometric Engine v4.0 Active"}
