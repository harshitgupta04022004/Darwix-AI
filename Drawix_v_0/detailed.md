# 📖 The Empathy Engine — Detailed Technical Documentation

This document provides an in-depth explanation of every module in The Empathy Engine, how they connect, the data flow between them, and the rationale behind each design decision.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Flow Pipeline](#2-data-flow-pipeline)
3. [Module Reference](#3-module-reference)
   - 3.1 [config.py — Configuration Hub](#31-configpy--configuration-hub)
   - 3.2 [emotion_detector.py — Emotion Classification](#32-emotion_detectorpy--emotion-classification)
   - 3.3 [ollama_client.py — Ollama API Integration](#33-ollama_clientpy--ollama-api-integration)
   - 3.4 [voice_modulator.py — Vocal Parameter Mapping](#34-voice_modulatorpy--vocal-parameter-mapping)
   - 3.5 [tts_engine.py — Text-to-Speech Orchestrator](#35-tts_enginepy--text-to-speech-orchestrator)
   - 3.6 [audio_processor.py — Audio Post-Processing](#36-audio_processorpy--audio-post-processing)
   - 3.7 [main.py — CLI Entry Point](#37-mainpy--cli-entry-point)
   - 3.8 [app.py — FastAPI Web Server](#38-apppy--fastapi-web-server)
   - 3.9 [static/index.html — Web Interface](#39-staticindexhtml--web-interface)
4. [How Files Connect — Dependency Graph](#4-how-files-connect--dependency-graph)
5. [Emotion-to-Voice Mapping Logic (The Math)](#5-emotion-to-voice-mapping-logic-the-math)
6. [Audio Processing Pipeline (Deep Dive)](#6-audio-processing-pipeline-deep-dive)
7. [API Specification](#7-api-specification)
8. [Configuration Reference](#8-configuration-reference)
9. [How to Add New Emotions](#9-how-to-add-new-emotions)
10. [How to Switch Between Providers](#10-how-to-switch-between-providers)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. System Architecture

The Empathy Engine follows a **modular pipeline architecture** where each module has a single responsibility and communicates through well-defined data structures (`EmotionResult` and `VocalParameters`).

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTRY POINTS                              │
│   ┌──────────┐    ┌──────────────────────────────────┐          │
│   │ main.py  │    │ app.py (FastAPI)                  │          │
│   │  (CLI)   │    │   GET /              → index.html │          │
│   │          │    │   POST /api/synthesize            │          │
│   └────┬─────┘    │   GET /api/audio/{file}           │          │
│        │          └────────────┬─────────────────────┘          │
│        │                       │                                 │
│        └───────────┬───────────┘                                 │
│                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           PROCESSING PIPELINE                            │    │
│  │                                                          │    │
│  │  ┌──────────────────┐   ┌───────────────────┐           │    │
│  │  │ emotion_detector │──▶│ voice_modulator   │           │    │
│  │  │   .py             │   │   .py              │           │    │
│  │  │                  │   │                   │           │    │
│  │  │ HuggingFace ─────│   │ EmotionResult ───▶│           │    │
│  │  │ Ollama ──────────│   │ VocalParameters   │           │    │
│  │  └──────────────────┘   └────────┬──────────┘           │    │
│  │                                   │                      │    │
│  │                                   ▼                      │    │
│  │  ┌──────────────────┐   ┌───────────────────┐           │    │
│  │  │ audio_processor  │◀──│ tts_engine        │           │    │
│  │  │   .py             │   │   .py              │           │    │
│  │  │                  │   │                   │           │    │
│  │  │ pydub            │   │ gTTS              │           │    │
│  │  │ (pitch/speed/vol)│   │ (base MP3)        │           │    │
│  │  └────────┬─────────┘   └───────────────────┘           │    │
│  │           │                                              │    │
│  │           ▼                                              │    │
│  │     output/*.wav                                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────────────────────────────────┐           │
│  │                  SUPPORT MODULES                  │           │
│  │  config.py ──── Data classes, constants, mapping  │           │
│  │  ollama_client.py ──── Ollama API calls           │           │
│  └──────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Pipeline

Here's exactly what happens when a user submits text:

### Step-by-Step Flow

```
1. USER INPUT
   "I just got promoted! This is amazing!" 
        │
        ▼
2. EMOTION DETECTION (emotion_detector.py)
   ├── Sends text to HuggingFace DistilBERT model
   ├── Model returns scores for all 6 emotions:
   │     joy: 0.9645, surprise: 0.0213, love: 0.0076, ...
   ├── Selects top emotion: "joy" (score: 0.9645)
   ├── Computes intensity: (0.9645 - 0.35) / (1.0 - 0.35) = 0.9454
   └── Returns: EmotionResult(label="joy", score=0.9645, intensity=0.9454)
        │
        ▼
3. VOICE MODULATION (voice_modulator.py)
   ├── Looks up "joy" in EMOTION_VOICE_MAP:
   │     rate_factor:     (1.10, 1.30)  → range
   │     pitch_semitones: (2.0, 4.0)    → range
   │     volume_db:       (2.0, 6.0)    → range
   ├── Interpolates with intensity=0.9454:
   │     rate  = 1.10 + (1.30 - 1.10) × 0.9454 = 1.289
   │     pitch = 2.0  + (4.0  - 2.0)  × 0.9454 = 3.89
   │     vol   = 2.0  + (6.0  - 2.0)  × 0.9454 = 5.78
   └── Returns: VocalParameters(rate=1.289, pitch=3.89, volume=5.78)
        │
        ▼
4. TTS GENERATION (tts_engine.py)
   ├── Creates gTTS object with text + language
   ├── Saves base audio to temp MP3 file
   ├── Calls audio_processor.process_audio()
   └── Cleans up temp file
        │
        ▼
5. AUDIO PROCESSING (audio_processor.py)
   ├── Loads MP3 via pydub (AudioSegment)
   ├── _apply_pitch(): Shift pitch up 3.89 semitones
   │     new_rate = 44100 × 2^(3.89/12) = 55,191 Hz
   │     Resample back to 44,100 Hz
   ├── _apply_speed(): Speed up by 1.289×
   │     new_rate = 44100 × 1.289 = 56,844 Hz
   │     Resample back to 44,100 Hz
   ├── _apply_volume(): Increase by +5.78 dB
   │     audio = audio + 5.78
   └── Export as .wav → output/empathy_joy_a1b2c3d4.wav
        │
        ▼
6. OUTPUT
   Returns path: /path/to/output/empathy_joy_a1b2c3d4.wav
```

---

## 3. Module Reference

### 3.1 config.py — Configuration Hub

**Purpose:** Single source of truth for all configurable parameters, data classes, and the emotion-to-voice mapping table.

**What's inside:**

| Item | Type | Description |
|------|------|-------------|
| `HUGGINGFACE_API_KEY` | `str/None` | Optional HF API key from env |
| `OLLAMA_API_URL` | `str` | Ollama server URL (default: `http://localhost:11434`) |
| `OLLAMA_MODEL` | `str` | Ollama model name (default: `llama3`) |
| `EMOTION_MODEL_NAME` | `str` | HF model identifier |
| `SUPPORTED_EMOTIONS` | `list` | Valid emotion labels |
| `CONFIDENCE_THRESHOLD` | `float` | Min confidence to classify (0.35) |
| `EmotionResult` | `dataclass` | Output of emotion detection |
| `VocalParameters` | `dataclass` | Output of voice modulation |
| `EMOTION_VOICE_MAP` | `dict` | Emotion → vocal parameter ranges |
| `OUTPUT_DIR` | `str` | Where generated audio files go |

**Why it exists:** Every module imports from `config.py` instead of hardcoding values. This makes the system easy to tune and extend — change a parameter in one place, and it propagates everywhere.

**Key Data Classes:**

```python
@dataclass
class EmotionResult:
    label: str              # "joy", "sadness", "anger", etc.
    score: float            # Raw confidence (0.0 - 1.0)
    intensity: float        # Normalized intensity (0.0 - 1.0)
    all_scores: dict        # All emotion scores

@dataclass
class VocalParameters:
    rate_factor: float      # Speed multiplier (1.0 = normal)
    pitch_semitones: float  # Pitch shift in semitones
    volume_db: float        # Volume change in dB
    emotion: str            # Source emotion
    intensity: float        # Source intensity
```

---

### 3.2 emotion_detector.py — Emotion Classification

**Purpose:** Analyze input text and classify it into one of 6 emotional categories + neutral.

**What this file does:**
- Provides a single public function: `detect_emotion(text, provider)`
- Supports two providers: `"huggingface"` (default) and `"ollama"`
- Lazily loads the HuggingFace model on first call (cached after that)
- Returns a standardized `EmotionResult` regardless of which provider is used

**HuggingFace Pipeline:**
- Model: `bhadresh-savani/distilbert-base-uncased-emotion`
- Fine-tuned on the **dair-ai/emotion** dataset
- Classifies into: `joy`, `sadness`, `anger`, `fear`, `surprise`, `love`
- Uses `top_k=None` to get scores for ALL emotions (not just the top one)
- Model is downloaded once (~260MB) and cached at `~/.cache/huggingface/`

**Intensity Calculation:**
```
intensity = (score - CONFIDENCE_THRESHOLD) / (1.0 - CONFIDENCE_THRESHOLD)
```
This maps the confidence score from `[0.35, 1.0]` → `[0.0, 1.0]`. Scores below 0.35 are treated as "neutral" (the model isn't confident enough to assign an emotion).

**Connections:**
- **Imports from:** `config.py` (model name, threshold, `EmotionResult`)
- **Used by:** `main.py`, `app.py`
- **Delegates to:** `ollama_client.py` (when provider="ollama")

---

### 3.3 ollama_client.py — Ollama API Integration

**Purpose:** Provide an alternative emotion detection backend using a locally running Ollama LLM.

**What this file does:**
- Sends a structured prompt to the Ollama `/api/generate` endpoint
- The prompt instructs the LLM to classify text into one of the supported emotions
- Parses the JSON response and returns a standardized `EmotionResult`
- Handles connection errors, timeouts, and malformed responses gracefully

**The Prompt:**
```
You are an emotion classification assistant.
Analyze the following text and classify its primary emotion into EXACTLY ONE of:
joy, sadness, anger, fear, surprise, love

Text to analyze: "{user_text}"

Respond with ONLY: {"emotion": "...", "confidence": 0.X}
```

**Why low temperature (0.1)?** We want deterministic, consistent classifications — not creative responses. A low temperature makes the LLM focus on the most likely answer.

**Connections:**
- **Imports from:** `config.py` (Ollama URL, model name, supported emotions)
- **Used by:** `emotion_detector.py` (when provider="ollama")

**How to set up Ollama:**
1. Install Ollama: `curl -fsSL https://ollama.ai/install.sh | sh`
2. Pull a model: `ollama pull llama3`
3. The server starts automatically at `http://localhost:11434`

---

### 3.4 voice_modulator.py — Vocal Parameter Mapping

**Purpose:** Convert an `EmotionResult` into concrete `VocalParameters` that describe how to modify the TTS audio.

**What this file does:**
- Looks up the detected emotion in `EMOTION_VOICE_MAP` (from config.py)
- Linearly interpolates each parameter between its min/max range using intensity
- Returns a `VocalParameters` object ready for the audio processor

**The Interpolation Formula:**
```
actual_value = min_value + (max_value - min_value) × intensity
```

**Example — Joy with intensity 0.7:**
```
rate_factor:     1.10 + (1.30 - 1.10) × 0.7 = 1.10 + 0.14 = 1.24x
pitch_semitones: 2.0  + (4.0  - 2.0)  × 0.7 = 2.0  + 1.4  = 3.4 semitones
volume_db:       2.0  + (6.0  - 2.0)  × 0.7 = 2.0  + 2.8  = 4.8 dB
```

**Additional function:** `describe_parameters(params)` generates a human-readable description like "Speaking 24% faster | Pitch raised by 3.4 semitones | Volume increased by 4.8 dB".

**Connections:**
- **Imports from:** `config.py` (mapping table, data classes)
- **Used by:** `main.py`, `app.py`
- **Produces output for:** `tts_engine.py`

---

### 3.5 tts_engine.py — Text-to-Speech Orchestrator

**Purpose:** Tie together TTS generation and audio post-processing into a single `synthesize()` call.

**What this file does:**
1. Takes `text` and `VocalParameters` as input
2. Generates a base (unmodulated) MP3 file using **gTTS** (Google Text-to-Speech)
3. Passes the MP3 to `audio_processor.py` for emotional modulation
4. Returns the path to the final `.wav` file
5. Cleans up temporary files

**Why gTTS (not pyttsx3)?**
- `pyttsx3` wraps platform-specific engines (eSpeak on Linux, SAPI5 on Windows)
- `pyttsx3` does NOT support reliable pitch control — `setProperty('pitch')` silently fails on most platforms
- `gTTS` uses Google's neural TTS API — much more natural-sounding base audio
- By generating a high-quality base and post-processing with `pydub`, we get full control over ALL three parameters (rate, pitch, volume)

**gTTS Details:**
- Outputs MP3 format (we convert to WAV via pydub)
- Requires internet connection (sends text to Google's servers)
- No API key needed (uses the free public endpoint)
- `slow=False` generates normal-speed speech as the base

**Connections:**
- **Imports from:** `config.py` (language, output dir), `audio_processor.py`
- **Used by:** `main.py`, `app.py`
- **Delegates to:** `audio_processor.py` for pitch/speed/volume modification

---

### 3.6 audio_processor.py — Audio Post-Processing

**Purpose:** Apply actual audio transformations (pitch shift, speed change, volume adjustment) to the TTS-generated audio.

**What this file does:**

#### Pitch Shifting (`_apply_pitch`)
Uses the frame-rate manipulation technique:
1. Calculate new sample rate: `new_rate = original_rate × 2^(semitones/12)`
2. Re-stamp the audio with the new frame rate (changes pitch AND speed)
3. Re-set to original frame rate (normalizes speed, keeps pitch change)

```
Example: Shift up 4 semitones from 24000 Hz
  new_rate = 24000 × 2^(4/12) = 24000 × 1.2599 = 30,238 Hz
  → Re-stamp as 30,238 Hz (sounds higher + shorter)
  → Re-set to 24000 Hz (sounds higher, normal speed)
```

#### Speed Change (`_apply_speed`)
Similar frame-rate technique:
1. Adjust frame rate by the rate factor
2. Re-set to original frame rate
3. Clamped to `[0.5, 2.0]` to prevent extreme distortion

#### Volume Adjustment (`_apply_volume`)
Simple dB addition/subtraction — `pydub` makes this trivial:
```python
louder_audio = audio + 5     # +5 dB
quieter_audio = audio - 3    # -3 dB
```

**Dependencies:**
- **pydub:** Python audio manipulation library
- **ffmpeg:** System dependency used by pydub for format conversion (MP3 → WAV, etc.)

**Connections:**
- **Imports from:** `config.py` (`VocalParameters` dataclass)
- **Used by:** `tts_engine.py`

---

### 3.7 main.py — CLI Entry Point

**Purpose:** Command-line interface for the Empathy Engine.

**What this file does:**
- Parses command-line arguments using `argparse`
- Supports `--text`, `--provider`, `--output`, `--verbose` flags
- Has an interactive mode (just run `python main.py` with no --text)
- Prints a formatted console output with emotion scores, vocal parameters, and file path
- Uses box-drawing characters and emoji for visual clarity

**Connections:**
- **Imports from:** `emotion_detector.py`, `voice_modulator.py`, `tts_engine.py`
- This is an entry point — nothing imports from it

---

### 3.8 app.py — FastAPI Web Server

**Purpose:** REST API and web UI server.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the web UI (`static/index.html`) |
| `POST` | `/api/synthesize` | Main endpoint — accepts `{text, provider}`, returns emotion analysis + audio URL |
| `GET` | `/api/audio/{filename}` | Serves generated audio files |
| `GET` | `/api/health` | Health check (`{"status": "ok"}`) |

**Request/Response Models (Pydantic):**

```python
# Request
class SynthesizeRequest:
    text: str           # Required
    provider: str       # "huggingface" (default) or "ollama"

# Response  
class SynthesizeResponse:
    emotion: { label: str, scores: dict }
    confidence: float
    intensity: float
    vocal_params: { rate_factor, pitch_semitones, volume_db, description }
    audio_url: str      # e.g. "/api/audio/empathy_joy_abc123.wav"
    audio_filename: str
```

**Connections:**
- **Imports from:** `emotion_detector.py`, `voice_modulator.py`, `tts_engine.py`, `config.py`
- **Serves:** `static/index.html`, `output/*.wav`

---

### 3.9 static/index.html — Web Interface

**Purpose:** Premium, single-page web application for interacting with the Empathy Engine.

**What this file does:**
- Provides a text input area with character count and sample emotion chips
- Provider toggle (HuggingFace / Ollama)
- Sends `POST /api/synthesize` requests to the backend
- Displays results: emotion badge with color, confidence/intensity values, score bar chart
- Shows vocal parameter gauges (rate, pitch, volume)
- Embedded HTML5 audio player with:
  - Play/pause button
  - Real-time waveform visualization (Web Audio API + Canvas)
  - Time display
  - Download button

**Design System:**
- **Font:** Inter (Google Fonts)
- **Theme:** Dark mode with glassmorphism
- **Colors:** Purple-pink gradient accent, per-emotion colors
- **Animations:** Floating icon, pop-in badges, animated score bars, shimmer button effect, smooth transitions

**No external dependencies** — pure HTML + CSS + vanilla JavaScript.

---

## 4. How Files Connect — Dependency Graph

```
                  ┌────────────┐
                  │  config.py │  ◄── Every module imports from here
                  └─────┬──────┘
                        │
          ┌─────────────┼─────────────────┐
          │             │                  │
          ▼             ▼                  ▼
  ┌──────────────┐ ┌──────────────┐ ┌───────────────┐
  │ emotion_     │ │ voice_       │ │ audio_        │
  │ detector.py  │ │ modulator.py │ │ processor.py  │
  └──────┬───────┘ └──────────────┘ └───────┬───────┘
         │                                   │
         │ (provider="ollama")               │
         ▼                                   │
  ┌──────────────┐                           │
  │ ollama_      │                           │
  │ client.py    │                           │
  └──────────────┘                           │
                                             │
  ┌──────────────────────────────────────────┘
  │
  ▼
  ┌──────────────┐
  │ tts_engine   │  ◄── Calls audio_processor.process_audio()
  │ .py          │
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│main.py │ │ app.py │  ◄── Both use the same pipeline
└────────┘ └────────┘
                │
                ▼
          ┌──────────┐
          │ static/  │
          │ index.   │
          │ html     │
          └──────────┘
```

**Import chain for a typical request:**

```python
# app.py imports:
from emotion_detector import detect_emotion   # Step 1
from voice_modulator import get_vocal_parameters  # Step 2
from tts_engine import synthesize             # Step 3

# tts_engine.py imports:
from audio_processor import process_audio     # Step 3b (sub-step)

# emotion_detector.py imports (conditionally):
from ollama_client import classify_emotion_with_ollama  # Only if provider="ollama"
```

---

## 5. Emotion-to-Voice Mapping Logic (The Math)

### The Emotional Modulation Table

Each emotion has a **range** for each vocal parameter. The actual value depends on the **intensity** of the emotion:

| Emotion | Rate Factor | Pitch (semitones) | Volume (dB) | Rationale |
|---------|-------------|-------------------|-------------|-----------|
| **Joy** | 1.10–1.30 | +2 to +4 | +2 to +6 | Happy speech is faster, higher-pitched, louder |
| **Sadness** | 0.78–0.88 | -4 to -2 | -6 to -3 | Sad speech is slow, low, quiet |
| **Anger** | 1.08–1.25 | +1 to +3 | +4 to +8 | Angry speech is fast and LOUD |
| **Fear** | 1.15–1.35 | +2 to +5 | -4 to -1 | Fearful speech is fast, high-pitched but quiet (whispering) |
| **Surprise** | 1.10–1.30 | +3 to +6 | +2 to +5 | Surprised speech is high-pitched (gasp effect) |
| **Love** | 0.88–0.95 | +0.5 to +2 | -2 to -0.5 | Loving speech is slow, gentle, soft |
| **Neutral** | 1.0 | 0 | 0 | No modulation |

### Intensity Scaling Formula

```
actual = min_val + (max_val - min_val) × intensity
```

Where `intensity ∈ [0.0, 1.0]` is derived from the classification confidence:

```
intensity = clamp((confidence - 0.35) / (1.0 - 0.35), 0, 1)
```

### Example Comparison

| Input Text | Emotion | Confidence | Intensity | Rate | Pitch | Volume |
|------------|---------|------------|-----------|------|-------|--------|
| "This is good." | joy | 0.62 | 0.42 | 1.18x | +2.8st | +3.7dB |
| "This is the BEST news EVER!!" | joy | 0.98 | 0.97 | 1.29x | +3.9st | +5.9dB |
| "I'm a bit down today." | sadness | 0.71 | 0.55 | 0.83x | -2.9st | -4.4dB |
| "Everything is terrible!" | sadness | 0.93 | 0.89 | 0.87x | -2.2st | -3.3dB |

---

## 6. Audio Processing Pipeline (Deep Dive)

### Pitch Shifting — How Frame-Rate Manipulation Works

When you play a 44,100 Hz audio file at 55,125 Hz, it sounds higher-pitched (like a chipmunk). To get ONLY the pitch change without speed change, we:

1. **Re-stamp** the audio with a new frame rate (e.g., 55,125 Hz)
   - pydub doesn't resample — it just changes the metadata
   - When played, it sounds higher AND faster
2. **Re-set** back to original frame rate (44,100 Hz)
   - pydub resamples the audio data
   - This stretches it back to normal duration
   - But the pitch shift remains!

```python
# The magic two lines:
pitched = audio._spawn(audio.raw_data, overrides={"frame_rate": new_rate})
pitched = pitched.set_frame_rate(original_rate)
```

### Speed Change — Same Technique, Different Purpose

Speed change uses the exact same trick, but with a simpler factor:

```python
new_rate = original_rate × speed_factor
```

### Volume — The Simplest One

pydub overloads the `+` and `-` operators for dB adjustment:

```python
louder = audio + 5.78     # Increase by 5.78 dB
quieter = audio - 3.2     # Decrease by 3.2 dB
```

### Processing Order

The order matters: **Pitch → Speed → Volume**

- Pitch is applied first because it modifies the raw audio data most significantly
- Speed is applied second (works on the already pitch-shifted audio)
- Volume is applied last (simple amplitude scaling, doesn't interact with timing/frequency)

---

## 7. API Specification

### POST /api/synthesize

**Request:**
```json
{
    "text": "I'm so excited about this project!",
    "provider": "huggingface"
}
```

**Response (200 OK):**
```json
{
    "emotion": {
        "label": "joy",
        "scores": {
            "joy": 0.9421,
            "surprise": 0.0312,
            "love": 0.0147,
            "anger": 0.0058,
            "fear": 0.0041,
            "sadness": 0.0021
        }
    },
    "confidence": 0.9421,
    "intensity": 0.9109,
    "vocal_params": {
        "rate_factor": 1.282,
        "pitch_semitones": 3.82,
        "volume_db": 5.64,
        "description": "Speaking 28% faster | Pitch raised by 3.8 semitones | Volume increased by 5.6 dB"
    },
    "audio_url": "/api/audio/empathy_joy_a1b2c3d4.wav",
    "audio_filename": "empathy_joy_a1b2c3d4.wav"
}
```

**Error Response (400/500):**
```json
{
    "detail": "Text input cannot be empty."
}
```

### GET /api/audio/{filename}

Returns the `.wav` file with `Content-Type: audio/wav`.

### GET /api/health

```json
{
    "status": "ok",
    "service": "The Empathy Engine",
    "version": "1.0.0"
}
```

---

## 8. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_API_KEY` | `None` | Optional. Not needed for local model usage. |
| `OLLAMA_API_URL` | `http://localhost:11434` | Base URL of your Ollama server. |
| `OLLAMA_MODEL` | `llama3` | Which Ollama model to use for emotion classification. |

### config.py Constants

| Constant | Value | Impact |
|----------|-------|--------|
| `CONFIDENCE_THRESHOLD` | 0.35 | Emotions below this confidence → "neutral" |
| `TTS_LANGUAGE` | "en" | Language for gTTS |
| `EMOTION_MODEL_NAME` | `bhadresh-savani/distilbert-base-uncased-emotion` | HuggingFace model |

---

## 9. How to Add New Emotions

To add a new emotion (e.g., "confused"):

### Step 1: Update `config.py`

```python
# Add to the list
SUPPORTED_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "love", "confused"]

# Add mapping entry
EMOTION_VOICE_MAP["confused"] = {
    "rate_factor":      (0.90, 0.95),    # Slightly slower
    "pitch_semitones":  (1.0,  3.0),     # Rising intonation
    "volume_db":        (-1.0, 0.0),     # Normal volume
}
```

### Step 2: Update `static/index.html`

Add the color for the new emotion in CSS:
```css
.emotion-badge.confused { 
    background: rgba(168, 162, 158, 0.15); 
    border-color: #a8a29e; 
    color: #a8a29e; 
}
.score-bar-fill.confused { background: #a8a29e; }
```

Add the emoji mapping in JavaScript:
```javascript
const EMOTION_EMOJIS = {
    // ... existing ones
    confused: '😕'
};
```

> **Note:** If you're using the HuggingFace model, it only supports the 6 emotions it was trained on. A new emotion would only work with Ollama (which can classify any emotion via its prompt). To support it with HuggingFace too, you'd need a different model or to fine-tune one.

---

## 10. How to Switch Between Providers

### Via CLI

```bash
# HuggingFace (default)
python main.py --text "Hello!" --provider huggingface

# Ollama
python main.py --text "Hello!" --provider ollama
```

### Via Web UI

Click the **🤗 HuggingFace** or **🦙 Ollama** toggle button in the input card.

### Via API

```json
{"text": "Hello!", "provider": "huggingface"}
{"text": "Hello!", "provider": "ollama"}
```

### Setting Ollama as Default

Edit the `.env` file:
```env
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

Then in your code, call:
```python
detect_emotion("Hello!", provider="ollama")
```

---

## 11. Troubleshooting

### "ffmpeg not found" Error
```
RuntimeError: Could not find ffmpeg
```
**Solution:** Install ffmpeg: `sudo apt-get install ffmpeg`

### HuggingFace Model Download is Slow
The first run downloads ~260MB. After that, it's cached at `~/.cache/huggingface/`.

### Ollama Connection Error
```
RuntimeError: Cannot connect to Ollama at http://localhost:11434
```
**Solution:** Start Ollama: `ollama serve` (in a separate terminal)

### gTTS Error (No Internet)
```
gTTS.tts.gTTSError: Connection error
```
**Solution:** gTTS requires internet. For offline TTS, you'd need to swap gTTS for pyttsx3 in `tts_engine.py`.

### Audio Sounds Distorted
If the pitch shift is too extreme, the audio quality degrades. The system clamps speed to `[0.5, 2.0]` and pitch shifts are moderate (max ~6 semitones), but very short texts may still sound artifacts.
