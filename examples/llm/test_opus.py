"""Test script for calling Claude Opus 4.6 via proxy endpoint."""

import requests
import json
import os

API_KEY = os.environ.get("A1X_API_KEY")
BASE_URL = "https://new.motchat.com"


def chat(prompt: str, model: str = "claude-opus-4-6") -> str | None:
    """Send a chat completion request."""
    url = f"{BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print("Testing Claude Opus 4.6 API...")
    answer = chat("如何赚钱")
    print(f"Response: {answer}")
