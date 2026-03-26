#!/usr/bin/env python3
"""
A1X Voice/Text Codegen — Generate and execute A1X arm control code via LLM.

Usage:
    python examples/ollama/a1x_voice_codegen.py
    python examples/ollama/a1x_voice_codegen.py --execute   # Auto-execute generated code
    python examples/ollama/a1x_voice_codegen.py --model claude-sonnet-4-20250514
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap

import readline  # noqa: F401 — enables arrow keys & line editing in input()
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("a1x_codegen")

# ---------------------------------------------------------------------------
# LLM API configuration (same proxy as test_opus.py)
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("A1X_API_KEY")
BASE_URL = "https://new.motchat.com"

# ---------------------------------------------------------------------------
# Load skill files as system prompt
# ---------------------------------------------------------------------------
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_skill_prompt() -> str:
    """Read SKILL.md and references/a1x-api.md, combine into a system prompt."""
    parts: list[str] = []
    for rel_path in ("SKILL.md", "references/a1x-api.md"):
        full_path = os.path.join(SKILL_DIR, rel_path)
        if os.path.exists(full_path):
            with open(full_path) as f:
                parts.append(f"# {rel_path}\n\n{f.read()}")
        else:
            logger.warning("Skill file not found: %s", full_path)
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = _load_skill_prompt()

# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def chat(
    user_prompt: str,
    history: list[dict[str, str]],
    model: str = "claude-opus-4-6",
) -> str | None:
    """Send a chat completion request with conversation history."""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error("LLM request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Code extraction & execution
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(response: str) -> str | None:
    """Extract the first Python code block from LLM response."""
    m = _CODE_BLOCK_RE.search(response)
    return m.group(1).strip() if m else None


def execute_code(code: str) -> None:
    """Write code to a temp file and execute it as a subprocess."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=project_root, delete=False, prefix="a1x_gen_"
    ) as f:
        f.write(code)
        tmp_path = f.name

    logger.info("Executing generated script: %s", tmp_path)
    try:
        subprocess.run(
            [sys.executable, tmp_path],
            cwd=project_root,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("Script exited with code %d", e.returncode)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A1X text-to-code arm control")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Automatically execute generated code (skip confirmation)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="LLM model name (default: claude-opus-4-6)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  A1X Arm Codegen — describe motions, get runnable code")
    print("=" * 60)
    print(f"  Model : {args.model}")
    print(f"  Execute: {'auto' if args.execute else 'ask'}")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'history' to view conversation history.")
    print("  Type 'clear' to reset conversation history.")
    print("=" * 60)

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if user_input.lower() == "history":
            if not history:
                print("(no history)")
            for i, msg in enumerate(history):
                tag = "You" if msg["role"] == "user" else "LLM"
                # Show full user messages, truncate LLM responses
                if msg["role"] == "user":
                    print(f"\n[{i+1}] [{tag}] {msg['content']}")
                else:
                    # Show first 200 chars of LLM response
                    text = msg["content"]
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"[{i+1}] [{tag}] {preview}")
            continue
        if user_input.lower() == "clear":
            history.clear()
            print("History cleared.")
            continue

        print("\n[LLM] Thinking...")
        response = chat(user_input, history, model=args.model)
        if response is None:
            print("[LLM] (no response)")
            continue

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # Display response
        print(f"\n[LLM]\n{response}")

        # Check for executable code
        code = extract_code(response)
        if code is None:
            continue

        print("\n" + "-" * 40)
        print("Executable code detected.")

        if args.execute:
            execute_code(code)
        else:
            try:
                ans = input("Run this code? [y/N] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSkipped.")
                continue
            if ans == "y":
                execute_code(code)
            else:
                print("Skipped.")


if __name__ == "__main__":
    main()
