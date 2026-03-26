#!/usr/bin/env python3
"""
A1X Grab — Code-as-Policies Grasping Orchestrator.

Interactive mode (default):
    python skills/a1x-grab-skill/scripts/a1x_grab.py
    > what's on the desk?
    > grab the yellow cube

Single-shot mode:
    python skills/a1x-grab-skill/scripts/a1x_grab.py "grab the yellow cube"
"""
from __future__ import annotations

import argparse
import logging
import re
import readline  # noqa: F401 — enables arrow keys & line editing in input()
import sys
from pathlib import Path

import os
import requests

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parents[3]        # A1Xsdk/
_SKILL_ROOT = _HERE.parents[1]         # a1x-grab-skill/
_SAM3_ROOT = PROJECT_ROOT / "refence_code" / "sam3"

for _p in (str(PROJECT_ROOT), str(_SAM3_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

sys.path.insert(0, str(_SKILL_ROOT))   # so `import robot_api` works
import robot_api

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("a1x_grab")

# ── Codegen LLM config ───────────────────────────────────────────────────────
_CLAUDE_API_KEY = os.environ.get("A1X_API_KEY", "")
_CLAUDE_BASE_URL = "https://new.motchat.com"
_CODEGEN_MODEL = "claude-opus-4-6"

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are a robot grasping assistant controlling a Galaxea A1X 6-axis robotic arm.

You will receive:
1. A scene description (what objects are visible and where)
2. A user message (may be in Chinese or English)

IMPORTANT: The scene description may use verbose names like "yellow rectangular paper note".
When you use these names in pick() or detect(), you MUST simplify them to short SAM3-friendly
names. Examples of simplification:
  "yellow rectangular paper note" → "yellow note"
  "small green plastic toy fish" → "green fish"
  "red ceramic coffee mug"      → "red mug"
  "yellow Y-shaped plastic piece" → "yellow piece"

## When the user asks a QUESTION (e.g. "what's on the desk?", "do you see a cup?"):
Reply with a helpful text answer based on the scene description.
Do NOT generate code for questions — just answer naturally.

## When the user gives an ACTION COMMAND:
Write a short Python program (under 40 lines) using ONLY these functions:

═══════════════════════════════════════════════════════
GRASPING FUNCTIONS
═══════════════════════════════════════════════════════

pick(object_name: str) -> bool
    Full pipeline: move to observe → capture RGBD → SAM3 detect → compute grasp → execute motion.
    Returns True on success, False on failure.
    ⚠️ object_name is fed directly to SAM3 vision model. Use SHORT, SIMPLE names:
      GOOD: "cup", "banana", "yellow cube", "red ball", "sponge", "note"
      BAD:  "yellow rectangular paper note", "small plastic toy shaped like a fish"
    Keep it to 1-3 words max: [color] + noun. Drop adjectives like "small",
    "rectangular", "plastic", "shaped like". SAM3 understands visual concepts,
    not long descriptions.

place() -> bool
    Moves to the fixed drop-off position and opens the gripper.
    Always call place() after a successful pick().

detect(object_name: str) -> dict | None
    Move to observation first, then call this. Captures a photo and runs SAM3.
    Returns {"bbox": [x1,y1,x2,y2], "score": float, "name": str} or None.
    Use in loops to check what objects remain after each pick+place cycle.
    ⚠️ Same naming rule as pick(): use SHORT names (1-3 words max).

═══════════════════════════════════════════════════════
ARM CONTROL FUNCTIONS
═══════════════════════════════════════════════════════

move_to_observation() -> bool
    Move arm to the observation position (where the camera can see the desk).
    Call this BEFORE detect() — detect needs the arm at observation to see objects.
    pick() calls this internally, so no need before pick().

move_to_home() -> bool
    Move arm to the home (rest) position.

move_ee_relative(dx=0.0, dy=0.0, dz=0.0) -> bool
    Move end-effector by a relative offset in meters.
    Coordinate system:  +X=forward  -X=backward  +Y=left  -Y=right  +Z=up  -Z=down
    Example: move_ee_relative(dy=-0.02) moves 2cm to the right.

get_ee_position() -> list[float]
    Returns current end-effector [x, y, z] in meters.

open_gripper() -> bool
    Open the gripper fully.

close_gripper() -> bool
    Close the gripper fully.

═══════════════════════════════════════════════════════
OTHER FUNCTIONS
═══════════════════════════════════════════════════════

ask_user(question: str) -> str
    Ask the user a clarifying question; returns their answer.

describe_scene(prompt: str = "...") -> str
    Capture a photo and describe it with VLM (slow). Only use when user asks.

═══════════════════════════════════════════════════════
RULES
═══════════════════════════════════════════════════════
- Use ONLY the functions above. No imports.
- Always call place() after a successful pick() — unless the user specifies
  a custom placement (e.g. "place it 2cm to the right"), in which case use
  move_ee_relative() + open_gripper() instead of place().
- For multiple targets, use a while loop:
  move_to_observation() → detect() → pick() → place() → repeat.
- If the user asks for ambiguous targets, call ask_user().
- Always call move_to_observation() at the END of the program so the arm
  returns to a known position and the camera can see the desk for the next task.
- End with exactly one print statement:
    print("Task completed successfully")   — if goal was achieved
    print("Task completed unsuccessfully") — if any step failed
- Wrap code in ```python ... ``` markers.

═══════════════════════════════════════════════════════
EXAMPLE 1: "pick up all the yellow objects"
═══════════════════════════════════════════════════════

```python
picked = 0
while True:
    move_to_observation()
    result = detect("yellow object")
    if result is None:
        break
    if pick(result["name"]):
        place()
        picked += 1
    else:
        break
move_to_observation()
if picked > 0:
    print("Task completed successfully")
else:
    print("Task completed unsuccessfully")
```

═══════════════════════════════════════════════════════
EXAMPLE 2: "grab the red cup and place it 3cm to the right"
═══════════════════════════════════════════════════════

```python
if pick("red cup"):
    move_ee_relative(dy=-0.03)
    open_gripper()
    move_to_observation()
    print("Task completed successfully")
else:
    print("Task completed unsuccessfully")
```
"""

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


# ── LLM interaction ──────────────────────────────────────────────────────────

def call_llm(
    scene_description: str,
    user_message: str,
    history: list[dict[str, str]],
    model: str = _CODEGEN_MODEL,
) -> str | None:
    """Send user message + scene to LLM with conversation history."""
    prompt = (
        f"Current scene description:\n{scene_description}\n\n"
        f"User: {user_message}"
    )

    url = f"{_CLAUDE_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {_CLAUDE_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 8192,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error("LLM request failed: %s", e)
        return None


def extract_code(response: str) -> str | None:
    """Extract Python code block from LLM response, if any."""
    m = _CODE_BLOCK_RE.search(response)
    return m.group(1).strip() if m else None


def save_generated_code(code: str) -> Path:
    """Save generated code to logs/generate_code/ with a timestamp filename."""
    from datetime import datetime

    log_dir = PROJECT_ROOT / "logs" / "generate_code"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = log_dir / f"{timestamp}.py"
    filepath.write_text(code)
    logger.info("Saved generated code to %s", filepath)
    return filepath


def execute_generated_code(code: str) -> None:
    """Run the generated code in a sandbox exposing only robot_api primitives."""
    sandbox = {
        # Grasping
        "describe_scene": robot_api.describe_scene,
        "detect": robot_api.detect,
        "pick": robot_api.pick,
        "place": robot_api.place,
        "ask_user": robot_api.ask_user,
        # Arm control
        "move_to_observation": robot_api.move_to_observation,
        "move_to_home": robot_api.move_to_home,
        "move_ee_relative": robot_api.move_ee_relative,
        "get_ee_position": robot_api.get_ee_position,
        "open_gripper": robot_api.open_gripper,
        "close_gripper": robot_api.close_gripper,
        # Builtins
        "print": print,
        "len": len,
        "range": range,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }
    exec(compile(code, "<generated>", "exec"), sandbox)


# ── Observation position ─────────────────────────────────────────────────────

def move_to_observation() -> None:
    """Move the arm to observation position for scene capture."""
    import a1x_control
    import time

    controller = a1x_control.JointController()
    time.sleep(1)
    if not controller.wait_for_joint_states(timeout=10):
        logger.warning("No joint state data, continuing anyway...")

    OBSERVATION = [0.0, 1.0, -0.93, 0.83, 0.0, 0.0]
    print("[Robot] Moving to observation position...")
    controller.move_to_position_smooth(
        OBSERVATION, steps=20, rate_hz=10.0, interpolation_type='cosine',
        wait_for_convergence=True,
    )
    print("[Robot] At observation position.")


def capture_scene() -> str:
    """Capture and describe the current scene."""
    print("[Scene] Capturing...")
    scene = robot_api.describe_scene()
    print(f"\n  Scene:\n{scene}\n")
    return scene


# ── Single-shot execution ────────────────────────────────────────────────────

def run_single(request: str, show_code: bool = False, model: str = _CODEGEN_MODEL) -> int:
    """Execute a single grasping request (backward-compatible CLI mode)."""
    print("=" * 60)
    print("  A1X Grab — Code as Policies")
    print("=" * 60)
    print(f"  Request: {request}")
    print()

    move_to_observation()

    scene = capture_scene()

    print("[LLM] Generating grasp code...")
    response = call_llm(scene, request, history=[], model=model)
    if response is None:
        print("Grasping task executed unsuccessfully — LLM failed")
        return 1

    code = extract_code(response)
    if code is None:
        # No code — LLM gave a text answer
        print(f"\n[LLM]\n{response}")
        return 0

    save_generated_code(code)

    if show_code:
        print("\n── Generated Code ──────────────────────────────────────")
        print(code)
        print("────────────────────────────────────────────────────────\n")

    print("[Exec] Running generated code...")
    try:
        execute_generated_code(code)
    except Exception as e:
        logger.error("Execution error: %s", e)
        print("Grasping task executed unsuccessfully")
        return 1

    return 0


# ── Interactive loop ──────────────────────────────────────────────────────────

def run_interactive(
    execute: bool = False,
    show_code: bool = False,
    model: str = _CODEGEN_MODEL,
) -> None:
    """Interactive REPL: ask questions or give grasping commands."""
    print("=" * 60)
    print("  A1X Grab — Interactive Mode")
    print("=" * 60)
    print(f"  Model  : {model}")
    print(f"  Execute: {'auto' if execute else 'ask'}")
    print("  Type 'quit' or 'exit' to stop.")
    print("  Type 'history' to view conversation history.")
    print("  Type 'clear' to reset conversation history.")
    print("  Type 'scene' to re-capture the scene description.")
    print("=" * 60)

    history: list[dict[str, str]] = []
    cached_scene: str | None = None
    has_moved_to_observation = False

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
                if msg["role"] == "user":
                    print(f"\n[{i+1}] [{tag}] {msg['content']}")
                else:
                    text = msg["content"]
                    preview = text[:200] + "..." if len(text) > 200 else text
                    print(f"[{i+1}] [{tag}] {preview}")
            continue

        if user_input.lower() == "clear":
            history.clear()
            cached_scene = None
            print("History and scene cache cleared.")
            continue

        if user_input.lower() == "scene":
            if not has_moved_to_observation:
                move_to_observation()
                has_moved_to_observation = True
            cached_scene = capture_scene()
            continue

        # ── First interaction: move to observation + capture scene ────
        if not has_moved_to_observation:
            move_to_observation()
            has_moved_to_observation = True

        if cached_scene is None:
            cached_scene = capture_scene()

        # ── Call LLM ──────────────────────────────────────────────────
        print("\n[LLM] Thinking...")
        response = call_llm(cached_scene, user_input, history, model=model)
        if response is None:
            print("[LLM] (no response)")
            continue

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        # ── Check if response contains code ───────────────────────────
        code = extract_code(response)

        if code is None:
            # Text-only answer (e.g. "what's on the desk?")
            print(f"\n[LLM]\n{response}")
            continue

        # Has code — show LLM explanation + code
        print(f"\n[LLM]\n{response}")
        save_generated_code(code)

        if show_code:
            print("\n── Extracted Code ──────────────────────────────────────")
            print(code)
            print("────────────────────────────────────────────────────────")

        # ── Execute ───────────────────────────────────────────────────
        if execute:
            print("\n[Exec] Running generated code...")
            try:
                execute_generated_code(code)
            except Exception as e:
                logger.error("Execution error: %s", e)
                print("Grasping task executed unsuccessfully")
        else:
            print("\n" + "-" * 40)
            print("Executable code detected.")
            try:
                ans = input("Run this code? [y/N] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nSkipped.")
                continue
            if ans == "y":
                print("\n[Exec] Running generated code...")
                try:
                    execute_generated_code(code)
                except Exception as e:
                    logger.error("Execution error: %s", e)
                    print("Grasping task executed unsuccessfully")
            else:
                print("Skipped.")

        # Invalidate scene cache after execution (objects may have moved)
        cached_scene = None


def main() -> None:
    parser = argparse.ArgumentParser(description="A1X Code-as-Policies grasp orchestrator")
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="Natural language grasping request (omit for interactive mode)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Auto-execute generated code without confirmation",
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Print generated code before executing",
    )
    parser.add_argument(
        "--model",
        default=_CODEGEN_MODEL,
        help=f"LLM model name (default: {_CODEGEN_MODEL})",
    )
    args = parser.parse_args()

    if args.request:
        sys.exit(run_single(args.request, show_code=args.show_code, model=args.model))
    else:
        run_interactive(
            execute=args.execute,
            show_code=args.show_code,
            model=args.model,
        )


if __name__ == "__main__":
    main()
