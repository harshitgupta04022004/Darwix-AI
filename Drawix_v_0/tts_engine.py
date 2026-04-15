"""
tts_engine.py — Text-to-Speech Engine Module

This module is the orchestrator that ties together text-to-speech generation
and audio post-processing. It:

  1. Takes input text and VocalParameters (from voice_modulator.py)
  2. Generates base speech audio using gTTS (Google Text-to-Speech)
  3. Passes the audio through audio_processor.py for emotional modulation
  4. Returns the path to the final .wav file

Data Flow:
    (text, VocalParameters)
      → gTTS generates base MP3
      → audio_processor applies pitch/speed/volume
      → final .wav file saved to output/

Connection to other modules:
    - Receives VocalParameters from voice_modulator.py
    - Uses audio_processor.py for post-processing
    - Called by main.py (CLI) and app.py (Web API)
    - Output directory defined in config.py

Dependencies:
    - gtts (pip install gtts) — Google Text-to-Speech
    - pydub + ffmpeg (via audio_processor.py)
"""

import logging
import os
import tempfile
import uuid

from gtts import gTTS

from config import TTS_LANGUAGE, OUTPUT_DIR, VocalParameters
from audio_processor import process_audio

logger = logging.getLogger(__name__)

# Module-level storage for the last generated SSML (for API access)
_last_ssml = [""]


def get_last_ssml() -> str:
    """Return the SSML text generated during the last synthesis call."""
    return _last_ssml[0]


def synthesize(
    text: str,
    vocal_params: VocalParameters,
    output_path: str = None,
) -> str:
    """
    Generate emotionally modulated speech audio from text.

    This is the main entry point for TTS generation. It:
      1. Creates a base MP3 using gTTS
      2. Applies emotional modulation via audio_processor
      3. Returns the path to the final .wav file

    Args:
        text:         The text to speak.
        vocal_params: Vocal modulation parameters (from voice_modulator.py).
        output_path:  Optional explicit output path. If None, a unique filename
                      is generated in the OUTPUT_DIR.

    Returns:
        Absolute path to the generated .wav audio file.
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text.")

    # Generate a unique filename if none provided
    if output_path is None:
        file_id = str(uuid.uuid4())[:8]
        emotion_slug = vocal_params.emotion or "neutral"
        output_path = os.path.join(OUTPUT_DIR, f"empathy_{emotion_slug}_{file_id}.wav")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Step 1: Apply SSML text transforms for enhanced expressiveness
    try:
        from ssml_processor import apply_ssml_transforms
        transformed_text, ssml_text = apply_ssml_transforms(text, vocal_params)
        # Store SSML for API access
        _last_ssml[0] = ssml_text
    except Exception as e:
        logger.warning(f"SSML processing skipped: {e}")
        transformed_text = text
        _last_ssml[0] = ""

    # Step 2: Generate base audio with gTTS (using transformed text)
    temp_mp3 = _generate_base_audio(transformed_text)

    try:
        # Step 3: Apply emotional modulation (pitch, speed, volume)
        final_path = process_audio(temp_mp3, vocal_params, output_path)
        logger.info(f"Synthesis complete: {final_path}")
        return final_path

    finally:
        # Clean up the temporary MP3 file
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            logger.debug(f"Cleaned up temp file: {temp_mp3}")


def _generate_base_audio(text: str) -> str:
    """
    Use gTTS to generate a base (unmodulated) MP3 audio file.

    gTTS sends the text to Google's TTS API and receives MP3 audio.
    The audio is saved to a temporary file for further processing.

    Args:
        text: The text to convert to speech.

    Returns:
        Path to the temporary MP3 file.
    """
    logger.info(f"Generating base TTS audio for: '{text[:80]}...'")

    # Create gTTS instance
    tts = gTTS(text=text, lang=TTS_LANGUAGE, slow=False)

    # Save to a temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp3", prefix="empathy_base_")
    os.close(temp_fd)  # Close the file descriptor — gTTS will open it itself

    tts.save(temp_path)

    logger.info(f"Base audio generated: {temp_path}")
    return temp_path


def get_output_filename(emotion: str) -> str:
    """
    Generate a descriptive output filename.

    Args:
        emotion: The detected emotion label.

    Returns:
        A unique filename string.
    """
    file_id = str(uuid.uuid4())[:8]
    return f"empathy_{emotion}_{file_id}.wav"
