import subprocess
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Professional AI De-Slopper & Stylometric API", version="6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    ai_text: str
    human_text: str

def run_detector_script(text_content: str):
    # حفظ النص مؤقتاً لفحصه بالسكريبت
    temp_filename = "temp_text.txt"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(text_content)
        
    try:
        # تشغيل سكريبت detect_ai_patterns.py واستخراج النتيجة بصيغة JSON
        result = subprocess.run(
            ["python3", "scripts/detect_ai_patterns.py", temp_filename, "--format", "json"],
            capture_output=True,
            text=True,
            check=True
        )
        data = json.loads(result.stdout)
        return data
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

@app.post("/compare")
def compare_texts(req: AnalysisRequest):
    if not req.ai_text.strip() or not req.human_text.strip():
        raise HTTPException(status_code=400, detail="Both texts are required.")
    
    ai_report = run_detector_script(req.ai_text)
    human_report = run_detector_script(req.human_text)
    
    return {
        "status": "success",
        "ai_metrics": ai_report,
        "human_metrics": human_report
    }

@app.get("/")
def root():
    return {"status": "active", "message": "507-Banned Pattern AI Detector API Live"}
