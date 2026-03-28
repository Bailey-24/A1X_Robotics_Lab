#!/usr/bin/env python3
"""
A1X Text-to-Speech — Convert text to audio via cloud TTS API.

Usage:
    # Speak text (plays immediately)
    python skills/a1x-tts/scripts/a1x_tts.py "Hello, I am the A1X robot"

    # Choose a voice
    python skills/a1x-tts/scripts/a1x_tts.py --voice nova "你好，我是A1X机器人"

    # Save to file instead of playing
    python skills/a1x-tts/scripts/a1x_tts.py --output /tmp/hello.mp3 "Hello world"

    # Use HD model for higher quality
    python skills/a1x-tts/scripts/a1x_tts.py --model tts-1-hd "High quality speech"

    # Interactive mode (type text to speak)
    python skills/a1x-tts/scripts/a1x_tts.py

    # Read from stdin (pipe text)
    echo "Hello from pipe" | python skills/a1x-tts/scripts/a1x_tts.py --stdin
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("a1x_tts")

# ── API config ────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("A1X_VLM_API_KEY", "")
BASE_URL = "https://api.chatanywhere.tech/v1"

VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
MODELS = ["tts-1", "tts-1-hd"]


def text_to_speech(
    text: str,
    voice: str = "onyx",
    model: str = "tts-1",
    output_path: str | None = None,
) -> Path:
    """Convert text to speech audio file.

    Args:
        text: The text to convert to speech.
        voice: Voice ID (alloy, echo, fable, onyx, nova, shimmer).
        model: TTS model (tts-1 or tts-1-hd).
        output_path: Path to save the audio file. If None, saves to a temp file.

    Returns:
        Path to the saved audio file.

    Raises:
        RuntimeError: If the API request fails.
    """
    if not API_KEY:
        raise RuntimeError(
            "A1X_VLM_API_KEY not set. Add to ~/.zshrc:\n"
            '  export A1X_VLM_API_KEY="your-key-here"'
        )

    url = f"{BASE_URL}/audio/speech"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
    }

    logger.info("TTS request: model=%s, voice=%s, text='%s'",
                model, voice, text[:50] + "..." if len(text) > 50 else text)

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, prefix="a1x_tts_")
        output_path = tmp.name
        tmp.close()

    path = Path(output_path)
    path.write_bytes(resp.content)
    logger.info("Audio saved to %s (%d bytes)", path, len(resp.content))
    return path


def play_audio(path: Path) -> None:
    """Play an audio file using available system player."""
    import subprocess
    import shutil

    players = ["mpv", "ffplay", "aplay", "paplay"]
    for player in players:
        if shutil.which(player):
            cmd = [player]
            if player == "mpv":
                cmd += ["--no-video", "--really-quiet"]
            elif player == "ffplay":
                cmd += ["-nodisp", "-autoexit", "-loglevel", "quiet"]
            cmd.append(str(path))
            logger.info("Playing with %s...", player)
            subprocess.run(cmd, check=True)
            return

    print(f"No audio player found. Audio saved to: {path}")
    print(f"Install one: sudo apt install mpv")


def speak(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    output_path: str | None = None,
    play: bool = True,
) -> Path:
    """Convert text to speech and optionally play it.

    This is the main API function for other scripts to import and use.

    Args:
        text: Text to speak.
        voice: Voice ID.
        model: TTS model.
        output_path: Save path (None for temp file).
        play: Whether to play the audio immediately.

    Returns:
        Path to the audio file.
    """
    path = text_to_speech(text, voice=voice, model=model, output_path=output_path)
    if play:
        play_audio(path)
    return path


def run_interactive(voice: str, model: str, output_dir: str | None) -> None:
    """Interactive mode: type text, hear it spoken."""
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    print("=" * 50)
    print("  A1X Text-to-Speech — Interactive Mode")
    print("=" * 50)
    print(f"  Model: {model}")
    print(f"  Voice: {voice}")
    print(f"  Voices: {', '.join(VOICES)}")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'voice <name>' to change voice.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n[TTS] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        # Voice switching command
        if user_input.lower().startswith("voice "):
            new_voice = user_input.split(None, 1)[1].strip().lower()
            if new_voice in VOICES:
                voice = new_voice
                print(f"Voice changed to: {voice}")
            else:
                print(f"Unknown voice '{new_voice}'. Available: {', '.join(VOICES)}")
            continue

        output_path = None
        if output_dir:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            output_path = str(Path(output_dir) / f"{ts}.mp3")

        try:
            speak(user_input, voice=voice, model=model, output_path=output_path)
        except Exception as e:
            logger.error("TTS failed: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="A1X Text-to-Speech")
    parser.add_argument(
        "text",
        nargs="?",
        default=None,
        help="Text to speak (omit for interactive mode)",
    )
    parser.add_argument(
        "--voice",
        default="onyx",
        choices=VOICES,
        help="Voice (default: onyx)",
    )
    parser.add_argument(
        "--model",
        default="tts-1",
        choices=MODELS,
        help="TTS model (default: tts-1)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save audio to this file path (default: play immediately)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Don't play audio, just save to file",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read text from stdin",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Save all audio files to this directory (interactive mode)",
    )
    args = parser.parse_args()

    # Stdin mode
    if args.stdin:
        text = sys.stdin.read().strip()
        if text:
            speak(text, voice=args.voice, model=args.model,
                  output_path=args.output, play=not args.no_play)
        return

    # Single-shot mode
    if args.text:
        speak(args.text, voice=args.voice, model=args.model,
              output_path=args.output, play=not args.no_play)
        return

    # Interactive mode
    run_interactive(voice=args.voice, model=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
