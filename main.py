from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
import joblib
import logging
import os
from typing import Literal
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Initialize FastAPI
app = FastAPI(
    title="Bangla Plagiarism Checker",
    description="Professional Bengali plagiarism detection service using TF-IDF and SBERT",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load NLTK data
try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error(f"NLTK data download failed: {str(e)}")
    raise RuntimeError("NLTK data initialization failed")

# Load stopwords
try:
    stopwords_df = pd.read_excel("bangla_stopwords.xlsx")
    bengali_stopwords = set(stopwords_df['word_list'].dropna().str.strip().str.lower())
    logger.info("Bengali stopwords loaded successfully")
except Exception as e:
    logger.error(f"Failed to load stopwords: {str(e)}")
    raise RuntimeError("Could not load stopwords")

# Load ML models
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    tfidf_matrix = np.load("tfidf_matrix.npy")
    sbert_model = SentenceTransformer("l3cube-pune/bengali-sentence-similarity-sbert")
    sbert_embeddings = np.load("sbert_embeddings.npy")
    logger.info("All ML models loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    raise RuntimeError("Could not load ML models")

# Pydantic models
class InputText(BaseModel):
    text: str
    min_similarity: float = 0.65  # Default threshold (0-1 scale)
    reference_text: str = ""  # Optional reference text

class DetectionResult(BaseModel):
    status: Literal["Plagiarized", "Original"]
    similarity_score: str          # As percentage (e.g., "87.34%")
    confidence: Literal["High", "Medium", "Low"]
    tfidf_score: str               # As percentage
    sbert_score: str               # As percentage
    processed_text: str
    threshold_used: str            # As percentage
    reference_comparison: str = "" # Similarity with reference text if provided

# Text processing functions
def preprocess_text(text: str) -> str:
    """Clean and normalize Bengali text"""
    try:
        # Remove numerals
        text = re.sub(r'[১-৯০]+', " ", text)
        text = re.sub(r'\s*\d+\s*', ' ', text)
        
        # Remove punctuation
        text = re.sub(u"['\"“”‘’]+|[.…-]+|[:;,.!?]+", " ", text)
        text = re.sub(r'[(){}[\]_]', ' ', text)
        text = re.sub(r'[\u200B-\u200D\uFEFF]', ' ', text)
        
        # Normalize whitespace
        return re.sub(r"\s+", " ", text).strip()
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise ValueError("Text preprocessing error")

def tokenize_and_filter(text: str) -> str:
    """Tokenize and remove stopwords"""
    try:
        words = word_tokenize(text)
        return " ".join(word for word in words if word.strip().lower() not in bengali_stopwords)
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        raise ValueError("Text tokenization error")

def to_percentage(score: float) -> str:
    """Convert 0-1 score to percentage string"""
    return f"{score * 100:.2f}%"

def get_confidence(score: float) -> Literal["High", "Medium", "Low"]:
    """Determine confidence level based on score"""
    if score >= 0.8: return "High"
    if score >= 0.5: return "Medium"
    return "Low"

# Serve the professional interface
@app.get("/", include_in_schema=False)
async def serve_interface(request: Request):
    """Serve the professional plagiarism checker interface"""
    return templates.TemplateResponse("index.html", {"request": request})

# API endpoint for plagiarism detection
@app.post("/api/check", response_model=DetectionResult, tags=["API"])
async def check_plagiarism(input: InputText):
    """
    Check for plagiarism in Bengali text
    
    Parameters:
    - text: Bengali text to analyze (required)
    - min_similarity: Threshold for plagiarism (default: 0.65)
    - reference_text: Optional text to compare against
    
    Returns:
    - Percentage-formatted similarity scores
    - Confidence level
    - Processed text
    - Reference comparison if provided
    """
    try:
        # Process main text
        cleaned = preprocess_text(input.text)
        filtered = tokenize_and_filter(cleaned)
        
        # Calculate similarities with corpus
        #tfidf_vec = vectorizer.transform([filtered])
        tfidf_vec = vectorizer.transform([input.text])
        tfidf_sim = cosine_similarity(tfidf_vec, tfidf_matrix).max()
        
        sbert_emb = sbert_model.encode(filtered, convert_to_tensor=True)
        sbert_sim = util.cos_sim(sbert_emb, sbert_embeddings).max().item()
        
        # Combine scores (70% TF-IDF + 30% SBERT)
        combined = 0.5 * tfidf_sim + 0.5 * sbert_sim
        
        # Handle reference text if provided
        reference_result = ""
        if input.reference_text.strip():
            ref_cleaned = preprocess_text(input.reference_text)
            ref_filtered = tokenize_and_filter(ref_cleaned)
            
            # Compare with reference
            ref_tfidf = vectorizer.transform([ref_filtered])
            ref_tfidf_sim = cosine_similarity(tfidf_vec, ref_tfidf).item()
            
            ref_sbert = sbert_model.encode(ref_filtered, convert_to_tensor=True)
            ref_sbert_sim = util.cos_sim(sbert_emb, ref_sbert).item()
            
            ref_combined = 0.5 * ref_tfidf_sim + 0.5 * ref_sbert_sim
            reference_result = to_percentage(ref_combined)
        
        # Determine result
        is_plagiarized = combined > input.min_similarity
        
        return {
            "status": "Plagiarized" if is_plagiarized else "Original",
            "similarity_score": to_percentage(combined),
            "confidence": get_confidence(combined),
            "tfidf_score": to_percentage(tfidf_sim),
            "sbert_score": to_percentage(sbert_sim),
            "processed_text": filtered,
            "threshold_used": to_percentage(input.min_similarity),
            "reference_comparison": reference_result
        }
        
    except ValueError as e:
        logger.error(f"Input error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal processing error")

import socket
import uvicorn

def get_free_port():
    s = socket.socket()
    s.bind(('', 0))
    return s.getsockname()[1]

if __name__ == "__main__":
    port = get_free_port()
    print(f"Running on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
