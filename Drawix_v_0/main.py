"""
main.py — CLI Entry Point for The Empathy Engine

This is the command-line interface for the Empathy Engine. It ties together
all modules into a simple workflow:

    1. Accept text input (via --text flag or interactive prompt)
    2. Detect emotion (via emotion_detector.py)
    3. Compute vocal parameters (via voice_modulator.py)
    4. Synthesize audio (via tts_engine.py)
    5. Output the result

Usage:
    python main.py --text "I'm so happy today!"
    python main.py --text "This is terrible news." --provider ollama
    python main.py --text "Hello there." --output my_audio.wav
    python main.py   (interactive mode — prompts for text input)
"""

import argparse
import logging
import sys

from config import EmotionResult, VocalParameters
from emotion_detector import detect_emotion
from voice_modulator import get_vocal_parameters, describe_parameters
from tts_engine import synthesize


def setup_logging(verbose: bool = False):
    """Configure logging for CLI output."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def print_banner():
    """Print the application banner."""
    banner = """
╔══════════════════════════════════════════════════════╗
║                                                      ║
║         🎭  THE EMPATHY ENGINE  🎭                   ║
║     AI-Powered Emotional Voice Synthesis              ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def print_results(text: str, emotion: EmotionResult, params: VocalParameters, audio_path: str):
    """Print formatted results to the console."""
    print("\n" + "─" * 55)
    print(f"  📝 Input Text:    \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
    print(f"  🎭 Emotion:       {emotion.label.upper()}")
    print(f"  📊 Confidence:    {emotion.score:.1%}")
    print(f"  🔥 Intensity:     {emotion.intensity:.1%}")

    if emotion.all_scores:
        print(f"  📈 All Scores:    ", end="")
        sorted_scores = sorted(emotion.all_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (label, score) in enumerate(sorted_scores):
            if i > 0:
                print(f"                    ", end="")
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            marker = " ◄" if label == emotion.label else ""
            print(f"{label:>10}: {bar} {score:.1%}{marker}")

    print(f"\n  🎤 Vocal Modulation:")
    print(f"     Rate:   {params.rate_factor:.2f}x  {'🔼' if params.rate_factor > 1 else '🔽' if params.rate_factor < 1 else '⏸'}")
    print(f"     Pitch:  {params.pitch_semitones:+.1f} semitones  {'🔼' if params.pitch_semitones > 0 else '🔽' if params.pitch_semitones < 0 else '⏸'}")
    print(f"     Volume: {params.volume_db:+.1f} dB  {'🔊' if params.volume_db > 0 else '🔉' if params.volume_db < 0 else '🔈'}")
    print(f"     📝 {describe_parameters(params)}")
    print(f"\n  🎵 Audio Output:  {audio_path}")
    print("─" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="🎭 The Empathy Engine — Emotionally Modulated Text-to-Speech",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --text "I'm thrilled about this!"
  python main.py --text "I'm so sad..." --provider ollama
  python main.py --text "Hello world" --output hello.wav
  python main.py -v --text "Test verbose mode"
        """,
    )

    parser.add_argument(
        "--text", "-t",
        type=str,
        help="The text to synthesize (if omitted, enters interactive mode)",
    )
    parser.add_argument(
        "--provider", "-p",
        type=str,
        choices=["huggingface", "ollama"],
        default="huggingface",
        help="Emotion detection provider (default: huggingface)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated in output/)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose/debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    print_banner()

    # Get text input
    text = args.text
    if not text:
        print("  Enter text to synthesize (or 'quit' to exit):\n")
        text = input("  > ").strip()
        if text.lower() in ("quit", "exit", "q"):
            print("  Goodbye! 👋\n")
            sys.exit(0)

    if not text:
        print("  ❌ No text provided. Exiting.\n")
        sys.exit(1)

    # Step 1: Detect emotion
    print(f"\n  ⏳ Detecting emotion using {args.provider}...")
    emotion_result = detect_emotion(text, provider=args.provider)

    # Step 2: Get vocal parameters
    vocal_params = get_vocal_parameters(emotion_result)

    # Step 3: Synthesize audio
    print(f"  ⏳ Synthesizing audio...")
    audio_path = synthesize(text, vocal_params, output_path=args.output)

    # Step 4: Display results
    print_results(text, emotion_result, vocal_params, audio_path)

    print("  ✅ Done! Play the audio file to hear the result.\n")


if __name__ == "__main__":
    main()
