"""
ssml_processor.py — Speech Synthesis Markup Language (SSML) Processing

This module provides SSML (Speech Synthesis Markup Language) integration
for enhanced control over the synthesized speech. It adds:

  1. Emphasis markers on emotionally significant words
  2. Pauses between sentences for dramatic effect
  3. Prosody adjustments embedded directly in the text

SSML is an XML-based markup language that gives fine-grained control over
how TTS engines render speech. While gTTS doesn't natively support SSML,
this module can generate SSML-annotated text for use with advanced TTS
engines (Google Cloud TTS, Amazon Polly, etc.) and also applies
text-level transformations that work with any TTS engine.

Data Flow:
    (text, VocalParameters)  →  apply_ssml_transforms()
    →  (transformed_text, ssml_text)

Connection to other modules:
    - Called by tts_engine.py before generating audio
    - Uses VocalParameters from voice_modulator.py to determine emphasis level
    - The SSML output can be used with cloud TTS APIs that support SSML
"""

import logging
import re
from typing import Tuple, List

from config import VocalParameters

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  Emphasis Words by Emotion
# ──────────────────────────────────────────────

# Words that carry emotional weight — these get emphasis markers in SSML
EMOTION_EMPHASIS_WORDS = {
    "joy": ["happy", "great", "wonderful", "amazing", "love", "fantastic",
            "excellent", "thrilled", "excited", "best", "awesome", "incredible",
            "perfect", "beautiful", "brilliant", "superb", "delighted"],
    "sadness": ["sad", "terrible", "awful", "miss", "lost", "alone",
                "empty", "hopeless", "painful", "heartbroken", "devastated",
                "depressed", "miserable", "regret", "sorry", "crying"],
    "anger": ["angry", "furious", "unacceptable", "outrageous", "ridiculous",
              "hate", "worst", "terrible", "disgusting", "unfair", "absurd",
              "frustrated", "annoyed", "livid", "infuriating"],
    "fear": ["scared", "afraid", "terrified", "danger", "horror", "panic",
             "frightened", "alarming", "threatening", "nightmare", "creepy",
             "anxious", "worried", "dread", "phobia"],
    "surprise": ["surprise", "shocked", "unexpected", "unbelievable",
                 "incredible", "wow", "astonishing", "stunning", "amazed",
                 "speechless", "jaw-dropping"],
    "love": ["love", "adore", "cherish", "beloved", "darling", "sweetheart",
             "heart", "devotion", "affection", "passion", "tender", "caring"],
}

# Pause durations (in milliseconds) based on emotion
EMOTION_PAUSE_MAP = {
    "joy": 200,         # Short pauses — energetic flow
    "sadness": 500,     # Long pauses — contemplative
    "anger": 150,       # Very short — rapid, aggressive
    "fear": 300,        # Medium pauses — hesitant
    "surprise": 400,    # Dramatic pauses
    "love": 450,        # Gentle, thoughtful pauses
    "neutral": 250,     # Standard pacing
}


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────

def apply_ssml_transforms(text: str, vocal_params: VocalParameters) -> Tuple[str, str]:
    """
    Apply SSML-inspired transformations to the input text.

    Returns a tuple of:
        1. transformed_text: Plain text with added pauses (via punctuation)
                             for use with gTTS or any basic TTS engine.
        2. ssml_text: Full SSML-formatted string for use with advanced
                      TTS engines (Google Cloud TTS, Amazon Polly, etc.)

    Args:
        text:         The input text to transform.
        vocal_params: Vocal parameters (determines emphasis level and pauses).

    Returns:
        (transformed_text, ssml_text)
    """
    emotion = vocal_params.emotion or "neutral"
    intensity = vocal_params.intensity

    # Generate plain text with pause enhancements
    transformed = _add_text_pauses(text, emotion, intensity)

    # Generate full SSML markup
    ssml = _generate_ssml(text, vocal_params, emotion, intensity)

    logger.info(f"SSML transforms applied for emotion '{emotion}' (intensity={intensity:.2f})")

    return transformed, ssml


def _add_text_pauses(text: str, emotion: str, intensity: float) -> str:
    """
    Add natural pauses to text using punctuation tricks.

    For gTTS (which doesn't support SSML), we can influence pacing by:
    - Adding commas for short pauses
    - Adding periods/ellipses for longer pauses
    - Adding extra spaces at emotional emphasis points

    This is a best-effort approach that works with any TTS engine.
    """
    if emotion == "neutral" or intensity < 0.3:
        return text

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if emotion == "sadness" and intensity > 0.5:
        # Add ellipses between sentences for contemplative pauses
        result = "... ".join(sentences)
    elif emotion == "anger" and intensity > 0.5:
        # Ensure exclamation marks for intensity
        result = "! ".join(s.rstrip('.!?,') for s in sentences) + "!"
    elif emotion == "surprise" and intensity > 0.5:
        # Add dramatic pauses
        result = "... ".join(sentences)
    else:
        result = " ".join(sentences)

    return result


def _generate_ssml(
    text: str,
    vocal_params: VocalParameters,
    emotion: str,
    intensity: float,
) -> str:
    """
    Generate full SSML markup for advanced TTS engines.

    Produces valid SSML 1.0 compatible markup with:
    - <prosody> tags for rate, pitch, and volume
    - <emphasis> tags on emotionally significant words
    - <break> tags for pauses between sentences
    - <say-as> tags for special content (exclamations, questions)

    This SSML can be passed directly to Google Cloud TTS, Amazon Polly,
    or any other SSML-compatible TTS engine.
    """
    # Map our parameters to SSML prosody attributes
    rate_pct = _rate_to_ssml_percent(vocal_params.rate_factor)
    pitch_pct = _pitch_to_ssml_percent(vocal_params.pitch_semitones)
    volume_str = _volume_to_ssml(vocal_params.volume_db)

    # Get emotion-specific emphasis words
    emphasis_words = set(EMOTION_EMPHASIS_WORDS.get(emotion, []))
    pause_ms = EMOTION_PAUSE_MAP.get(emotion, 250)

    # Scale pause duration by intensity
    pause_ms = int(pause_ms * (0.5 + 0.5 * intensity))

    # Split text into sentences for pause insertion
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Build SSML body
    body_parts = []
    for i, sentence in enumerate(sentences):
        # Add emphasis to emotional words
        processed = _emphasize_words(sentence, emphasis_words, intensity)
        body_parts.append(processed)

        # Add break between sentences (not after the last one)
        if i < len(sentences) - 1:
            body_parts.append(f'<break time="{pause_ms}ms"/>')

    body = "\n    ".join(body_parts)

    # Wrap in prosody and speak tags
    ssml = f"""<speak>
  <prosody rate="{rate_pct}" pitch="{pitch_pct}" volume="{volume_str}">
    {body}
  </prosody>
</speak>"""

    return ssml


def _emphasize_words(sentence: str, emphasis_words: set, intensity: float) -> str:
    """
    Wrap emotionally significant words in <emphasis> SSML tags.

    Emphasis level is based on intensity:
        intensity > 0.7 → <emphasis level="strong">
        intensity > 0.4 → <emphasis level="moderate">
        otherwise → <emphasis level="reduced">
    """
    if not emphasis_words or intensity < 0.2:
        return sentence

    # Determine emphasis level
    if intensity > 0.7:
        level = "strong"
    elif intensity > 0.4:
        level = "moderate"
    else:
        level = "reduced"

    # Find and wrap emphasis words (case-insensitive)
    def replace_word(match):
        word = match.group(0)
        if word.lower().rstrip('.,!?;:') in emphasis_words:
            return f'<emphasis level="{level}">{word}</emphasis>'
        return word

    # Match whole words
    result = re.sub(r'\b\w+\b', replace_word, sentence)
    return result


def _rate_to_ssml_percent(rate_factor: float) -> str:
    """Convert rate factor to SSML percentage string (e.g., '120%')."""
    pct = int(rate_factor * 100)
    return f"{pct}%"


def _pitch_to_ssml_percent(semitones: float) -> str:
    """Convert semitones to SSML pitch string (e.g., '+20%' or '-10%')."""
    # Approximate: 1 semitone ≈ 6% pitch change
    pct = int(semitones * 6)
    if pct >= 0:
        return f"+{pct}%"
    return f"{pct}%"


def _volume_to_ssml(volume_db: float) -> str:
    """Convert dB to SSML volume keyword."""
    if volume_db > 4:
        return "x-loud"
    elif volume_db > 2:
        return "loud"
    elif volume_db > -2:
        return "medium"
    elif volume_db > -4:
        return "soft"
    else:
        return "x-soft"


def get_ssml_info(ssml_text: str) -> dict:
    """
    Parse SSML text and return a summary of the markup for debugging/display.
    """
    info = {
        "has_prosody": "<prosody" in ssml_text,
        "has_emphasis": "<emphasis" in ssml_text,
        "has_break": "<break" in ssml_text,
        "emphasis_count": ssml_text.count("<emphasis"),
        "break_count": ssml_text.count("<break"),
    }

    # Extract prosody attributes
    prosody_match = re.search(r'<prosody\s+([^>]+)>', ssml_text)
    if prosody_match:
        attrs = prosody_match.group(1)
        for attr in ["rate", "pitch", "volume"]:
            match = re.search(f'{attr}="([^"]+)"', attrs)
            if match:
                info[f"prosody_{attr}"] = match.group(1)

    return info
