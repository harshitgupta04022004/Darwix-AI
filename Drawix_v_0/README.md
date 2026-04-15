# 🎭 The Empathy Engine

**AI-Powered Emotional Voice Synthesis** — Detect emotion from text and generate expressive, human-like speech that conveys the right feeling.

> Moving beyond monotonic delivery to achieve emotional resonance.

---

## 🌟 What It Does

The Empathy Engine is a modular Python service that dynamically modulates the vocal characteristics of synthesized speech based on the detected emotion of the source text. It bridges the gap between text-based sentiment and expressive audio output.

### Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **6 Granular Emotions** | Joy, Sadness, Anger, Fear, Surprise, Love (+ Neutral fallback) |
| 🔥 **Intensity Scaling** | Confidence score scales the degree of vocal modulation |
| 🎤 **3 Vocal Parameters** | Rate (speed), Pitch (semitones), Volume (dB) |
| 🤗 **HuggingFace Model** | DistilBERT fine-tuned on emotion dataset (runs locally) |
| 🦙 **Ollama Fallback** | Use any local LLM for emotion classification |
| 🌐 **Premium Web UI** | Dark-mode glassmorphism interface with waveform visualizer |
| 💻 **CLI Interface** | Full command-line interface with formatted output |
| 📝 **SSML Integration** | Generates Speech Synthesis Markup Language for emphasis, pauses & prosody |

---

## 📋 Prerequisites

- **Python 3.9+**
- **ffmpeg** (required by pydub for audio processing)
- **Internet connection** (for gTTS and first-time model download)
- *Optional:* [Ollama](https://ollama.ai/) running locally (if using Ollama provider)

### Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg
```

---

## 🚀 Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Drawix
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment (Optional)

Create a `.env` file for custom configuration:

```env
# Optional: HuggingFace API key (not needed for local model usage)
HUGGINGFACE_API_KEY=hf_your_key_here

# Optional: Ollama configuration (defaults shown)
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

---

## 🖥️ Usage

### CLI Mode

```bash
# Basic usage
python main.py --text "I'm so happy today!"

# Use Ollama provider
python main.py --text "This is terrible." --provider ollama

# Custom output path
python main.py --text "Hello world!" --output my_audio.wav

# Verbose mode
python main.py -v --text "I'm scared of the dark"

# Interactive mode (no --text flag)
python main.py
```

### Web Interface

```bash
# Start the web server
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

### API Endpoint

```bash
curl -X POST http://localhost:8000/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "I am thrilled!", "provider": "huggingface"}'
```

---

## 🏗️ Architecture

```
Text Input → Emotion Detection → Voice Modulation → TTS + Audio Processing → .wav Output
```

See **[detailed.md](detailed.md)** for in-depth documentation of every module and how they connect.

---

## 📁 Project Structure

```
├── app.py                  # FastAPI web server
├── main.py                 # CLI entry point
├── config.py               # Configuration & constants
├── emotion_detector.py     # Emotion classification (HuggingFace/Ollama)
├── ollama_client.py        # Ollama API client
├── voice_modulator.py      # Emotion → vocal parameter mapping
├── tts_engine.py           # Text-to-speech orchestrator
├── audio_processor.py      # Audio post-processing (pitch/speed/volume)
├── ssml_processor.py       # SSML generation (emphasis, pauses, prosody)
├── requirements.txt        # Python dependencies
├── detailed.md             # Comprehensive module documentation
├── static/
│   └── index.html          # Web UI
└── output/                 # Generated audio files
```

---

## 🎨 Design Choices

### Why gTTS + pydub (not pyttsx3)?
- **pyttsx3** has limited pitch control across platforms (no reliable `setProperty('pitch')` on Linux/eSpeak).
- **gTTS** provides natural-sounding Google TTS as a base, and **pydub** gives us full programmatic control over pitch (frame-rate manipulation), speed, and volume post-generation.

### Why intensity scaling?
- A flat emotion→voice mapping makes "This is good" sound exactly like "THIS IS THE BEST NEWS EVER!" — that's unrealistic.
- By using the model's confidence score as an intensity signal, we get proportional modulation that feels natural.

### Why HuggingFace + Ollama dual support?
- **HuggingFace**: Fast, accurate, runs offline after first download. Best for production.
- **Ollama**: Flexible, uses any LLM you have locally. Great for experimentation and custom prompts.

### SSML Integration
- The engine generates **Speech Synthesis Markup Language** (SSML) for every synthesis request.
- SSML adds `<emphasis>` tags on emotionally significant words, `<break>` tags for emotion-appropriate pauses, and `<prosody>` tags for rate/pitch/volume.
- While gTTS doesn't natively consume SSML, the SSML output is available via the API (`ssml_text` field) for use with SSML-compatible engines (Google Cloud TTS, Amazon Polly).
- Text-level transforms (pause insertion via punctuation) ARE applied to the gTTS input.

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
