# A1X LLM Codegen

Use natural language to control the A1X robotic arm. An LLM translates your text commands into Python control scripts and executes them on the real robot.

## Prerequisites

```bash
# 1. Configure CAN bus (required after every reboot)
sudo ip link set can0 type can bitrate 1000000 sample-point 0.875 dbitrate 5000000 fd on dsample-point 0.875
sudo ip link set up can0

# 2. Activate environment
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
```

## Quick Start

### Interactive mode (type commands one by one)

```bash
python examples/llm/a1x_text_codegen.py --execute
```

Then type your command at the `[You] >` prompt. Type `quit` to exit.

### One-shot mode (pipe a single command)

```bash
printf 'your command here\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

### Multi-command mode (pipe several commands in sequence)

```bash
printf 'command 1\ncommand 2\ncommand 3\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

## Examples

### Joint control (named positions)

```bash
# Go to home position
printf '回到home位置\nquit\n' | python examples/llm/a1x_text_codegen.py --execute

# Go to observation position
printf '去观测位置\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

### End-effector control (Cartesian movement)

```bash
# Move to observation, then forward 2cm
printf '去观测位置，然后向前移动2厘米\nquit\n' | python examples/llm/a1x_text_codegen.py --execute

# Move up 3cm
printf '向上移动3厘米\nquit\n' | python examples/llm/a1x_text_codegen.py --execute

# Move left 1cm and forward 2cm
printf '向左移动1厘米，同时向前移动2厘米\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

### Combined sequences

```bash
# Observation → forward 2cm → up 3cm
printf '去观测位置，然后向前移动2厘米，然后向上移动3厘米\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

### English commands work too

```bash
printf 'Move to the observation position, then move forward 2 centimeters\nquit\n' | python examples/llm/a1x_text_codegen.py --execute
```

## Direction Reference

| Direction | Axis | Example command |
|-----------|------|-----------------|
| forward   | +X   | "向前移动2厘米" / "move forward 2cm" |
| backward  | -X   | "向后移动1厘米" / "move backward 1cm" |
| left      | +Y   | "向左移动3厘米" / "move left 3cm" |
| right     | -Y   | "向右移动1厘米" / "move right 1cm" |
| up        | +Z   | "向上移动2厘米" / "move up 2cm" |
| down      | -Z   | "向下移动1厘米" / "move down 1cm" |

## Flags

| Flag | Description |
|------|-------------|
| `--execute` | Auto-execute generated code (skip confirmation prompt) |
| `--model MODEL` | LLM model name (default: `claude-opus-4-6`) |

Without `--execute`, the script shows the generated code and asks `Run this code? [y/N]` before executing.

## Interactive Commands

| Command | Description |
|---------|-------------|
| `quit` / `exit` | Exit the program |
| `history` | Show conversation history |
| `clear` | Reset conversation history |
