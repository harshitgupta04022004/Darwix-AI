"""
app.py — FastAPI Web Server for The Empathy Engine

This module provides a web interface and REST API for the Empathy Engine.
It serves a premium single-page web application and handles audio synthesis
requests via API endpoints.

Endpoints:
    GET  /                    → Serves the web UI (static/index.html)
    POST /api/synthesize      → Accepts text + provider, returns emotion analysis + audio URL
    GET  /api/audio/{filename} → Serves generated audio files from the output/ directory

Connection to other modules:
    - Uses emotion_detector.py for text analysis
    - Uses voice_modulator.py for parameter computation
    - Uses tts_engine.py for audio generation
    - Reads configuration from config.py

Running:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import OUTPUT_DIR, SERVER_HOST, SERVER_PORT
from emotion_detector import detect_emotion
from voice_modulator import get_vocal_parameters, describe_parameters
from tts_engine import synthesize, get_last_ssml

# ──────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────

app = FastAPI(
    title="The Empathy Engine",
    description="AI-Powered Emotional Voice Synthesis — detect emotion from text and generate expressive speech.",
    version="1.0.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# ──────────────────────────────────────────────
#  Request / Response Models
# ──────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    text: str
    provider: str = "huggingface"  # "huggingface" or "ollama"


class EmotionScores(BaseModel):
    label: str
    scores: dict


class VocalParams(BaseModel):
    rate_factor: float
    pitch_semitones: float
    volume_db: float
    description: str


class SynthesizeResponse(BaseModel):
    emotion: EmotionScores
    confidence: float
    intensity: float
    vocal_params: VocalParams
    audio_url: str
    audio_filename: str
    ssml_text: str = ""


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web interface."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Web UI not found. Ensure static/index.html exists.")
    
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/synthesize", response_model=SynthesizeResponse)
async def api_synthesize(request: SynthesizeRequest):
    """
    Main API endpoint: analyze text emotion and generate modulated speech.

    Accepts:
        { "text": "I'm so happy!", "provider": "huggingface" }

    Returns:
        {
            "emotion": { "label": "joy", "scores": {...} },
            "confidence": 0.98,
            "intensity": 0.97,
            "vocal_params": { "rate_factor": 1.28, ... },
            "audio_url": "/api/audio/empathy_joy_abc123.wav",
            "audio_filename": "empathy_joy_abc123.wav"
        }
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")

    if len(text) > 5000:
        raise HTTPException(status_code=400, detail="Text input exceeds 5000 character limit.")

    provider = request.provider.lower()
    if provider not in ("huggingface", "ollama"):
        raise HTTPException(status_code=400, detail=f"Invalid provider '{provider}'. Use 'huggingface' or 'ollama'.")

    try:
        # Step 1: Detect emotion
        logger.info(f"API request — text: '{text[:50]}...', provider: {provider}")
        emotion_result = detect_emotion(text, provider=provider)

        # Step 2: Get vocal parameters
        vocal_params = get_vocal_parameters(emotion_result)

        # Step 3: Synthesize audio
        audio_path = synthesize(text, vocal_params)
        audio_filename = os.path.basename(audio_path)

        # Get the generated SSML text
        ssml_text = get_last_ssml()

        # Build response
        response = SynthesizeResponse(
            emotion=EmotionScores(
                label=emotion_result.label,
                scores=emotion_result.all_scores,
            ),
            confidence=emotion_result.score,
            intensity=emotion_result.intensity,
            vocal_params=VocalParams(
                rate_factor=vocal_params.rate_factor,
                pitch_semitones=vocal_params.pitch_semitones,
                volume_db=vocal_params.volume_db,
                description=describe_parameters(vocal_params),
            ),
            audio_url=f"/api/audio/{audio_filename}",
            audio_filename=audio_filename,
            ssml_text=ssml_text,
        )

        logger.info(f"Synthesis complete — emotion: {emotion_result.label}, file: {audio_filename}")
        return response

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


@app.get("/api/audio/{filename}")
async def serve_audio(filename: str):
    """Serve a generated audio file from the output directory."""
    # Sanitize filename to prevent directory traversal
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Audio file '{safe_filename}' not found.")

    return FileResponse(
        path=file_path,
        media_type="audio/wav",
        filename=safe_filename,
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "The Empathy Engine", "version": "1.0.0"}


# ──────────────────────────────────────────────
#  Run with: uvicorn app:app --reload
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=SERVER_HOST, port=SERVER_PORT, reload=True)
