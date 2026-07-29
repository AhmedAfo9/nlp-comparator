import spacy
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import re
import os

app = FastAPI(title="31-Pattern AI Stylometric Engine", version="5.0")

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

AI_PATTERN_RULES = {
    "significance_inflation": r"\b(pivotal role|testament to|paramount importance|profound impact|crucial step)\b",
    "notability_namedropping": r"\b(as widely acknowledged|noted by experts|renowned scholars)\b",
    "superficial_analysis": r"\b(multifaceted approach|nuanced perspective|complex interplay|deep dive)\b",
    "promotional_language": r"\b(groundbreaking|unrivaled|seamlessly|game-changer|unmatched)\b",
    "vague_attributions": r"\b(studies suggest|experts believe|it is widely accepted|scholars argue)\b",
    "formulaic_outlook": r"\b(challenges and future outlook|horizon|looking ahead, it is clear)\b",
    "ai_vocabulary_overuse": r"\b(delve|tapestry|beacon|foster|interplay|underscores|intricate|ethereal|vibrant)\b",
    "copula_avoidance": r"\b(utilization of|implementation of|facilitation of|conduct an analysis of)\b",
    "negative_parallelisms": r"\b(not (only|merely) .+ but (also|rather))\b",
    "rule_of_three": r"\b(\w+,\s+\w+,\s+and\s+\w+)\b",
    "false_ranges": r"\b(ranging from .+ to .+ and beyond)\b",
    "em_dash_overuse": r"—",
    "boldface_overuse": r"\*\*[^*]+\*\*",
    "inline_header_lists": r"^\s*[-*•]?\s*\*\*[^*]+\*\*:",
    "chat_artifacts": r"\b(certainly!|here is a|as an ai language model|in conclusion,)\b",
    "filler_openers": r"\b(it is worth (noting|highlighting) that|in order to|it goes without saying)\b",
    "excessive_hedging": r"\b(could potentially be|might suggest that|it appears that it may)\b",
    "throat_clearing": r"\b(in today's (digital|fast-paced|modern) world|since the dawn of)\b",
    "meta_discourse": r"\b(this section will|having examined|we now turn to|as discussed above)\b",
    "abstract_reification": r"\b(embarking on a journey|roadmap towards|navigating the landscape)\b",
    "moralizing_ends": r"\b(only time will tell|a brighter future|serves as a reminder for humanity)\b"
}

def scan_31_ai_patterns(text: str):
    detected_patterns = []
    total_matches = 0
    
    for pattern_name, regex_expr in AI_PATTERN_RULES.items():
        matches = re.findall(regex_expr, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            count = len(matches)
            total_matches += count
            clean_name = pattern_name.replace("_", " ").title()
            detected_patterns.append({
                "pattern": clean_name,
                "count": count,
                "samples": list(set([m if isinstance(m, str) else m[0] for m in matches]))[:3]
            })
            
    return detected_patterns, total_matches

def compute_deep_nlp_metrics(text: str):
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_punct and not t.is_space]
    if not tokens:
        return {}
    
    words = [t.text.lower() for t in tokens]
    unique_words = set(words)
    ttr = (len(unique_words) / len(words)) * 100 if words else 0
    
    sentences = list(doc.sents)
    sent_lengths = [len([t for t in sent if not t.is_punct and not t.is_space]) for sent in sentences]
    
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_variance = float(np.var(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    
    pos_counts = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for t in tokens:
        if t.pos_ in pos_counts:
            pos_counts[t.pos_] += 1
            
    pos_ratios = {k: round((v / len(tokens)) * 100, 2) for k, v in pos_counts.items()}
    patterns_detected, total_slop_count = scan_31_ai_patterns(text)

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "lexical_diversity_ttr": round(ttr, 2),
        "avg_sentence_length": round(avg_sent_len, 2),
        "burstiness_variance": round(sent_variance, 2),
        "pos_distribution": pos_ratios,
        "ai_patterns": patterns_detected,
        "ai_slop_score": total_slop_count
    }

def generate_comparative_critique(ai, hum):
    reasons = []
    
    if hum["burstiness_variance"] > ai["burstiness_variance"]:
        reasons.append(f"Rhythmic Burstiness: Human reference demonstrates natural clause variance ({hum['burstiness_variance']} vs {ai['burstiness_variance']}), confirming dynamic sentence structure.")
    else:
        reasons.append(f"Rhythmic Monotony: AI sample shows low structural variance ({ai['burstiness_variance']}), indicating machine token smoothing.")
        
    if ai["ai_slop_score"] > hum["ai_slop_score"]:
        patterns_list = [p['pattern'] for p in ai['ai_patterns'][:3]]
        reasons.append(f"AI Signature Patterns: AI text triggered {ai['ai_slop_score']} rule violations across categories including [{', '.join(patterns_list)}].")
    
    if ai["pos_distribution"]["ADJ"] > hum["pos_distribution"]["ADJ"]:
        reasons.append(f"Stylistic Inflation: AI text over-relies on qualifying adjectives ({ai['pos_distribution']['ADJ']}% vs {hum['pos_distribution']['ADJ']}%).")
        
    return reasons

@app.post("/compare")
def compare_texts(req: AnalysisRequest):
    if not req.ai_text.strip() or not req.human_text.strip():
        raise HTTPException(status_code=400, detail="Both texts are required.")
    
    ai_metrics = compute_deep_nlp_metrics(req.ai_text)
    human_metrics = compute_deep_nlp_metrics(req.human_text)
    critique = generate_comparative_critique(ai_metrics, human_metrics)
    
    return {
        "status": "success",
        "ai_metrics": ai_metrics,
        "human_metrics": human_metrics,
        "diagnostic_verdict": critique
    }

@app.get("/")
def root():
    return {"status": "active", "message": "31-Pattern AI Engine Live"}
