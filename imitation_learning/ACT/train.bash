#!/bin/bash
# =============================================================================
# ACT Training Script for A1X Robot
# =============================================================================
# This script trains an ACT (Action Chunking Transformers) policy using LeRobot
# on data collected from the A1X robotic arm.
#
# Usage:
#   ./train.bash                           # Use default settings (5 epochs)
#   ./train.bash --dataset a1x/my_task     # Use custom dataset
#   ./train.bash --epochs 10               # Train for specific epochs
#   ./train.bash --steps 50000             # Train for specific steps (overrides epochs)
#   ./train.bash --gpu 0                   # Use specific GPU
#
# Prerequisites:
#   1. Conda environment: conda activate a1x_ros
#   2. LeRobot installed: pip install -e refence_code/lerobot
#   3. Dataset collected and converted to LeRobot format
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration (can be overridden by command line arguments)
# =============================================================================

# Dataset settings
DATASET_REPO_ID="${DATASET_REPO_ID:-a1x/yoloe_grasp_white_object}"
DATASET_ROOT="${DATASET_ROOT:-$HOME/.cache/huggingface/lerobot}"

# Training settings
EPOCHS="${EPOCHS:-5}"                           # Number of epochs (default: 5)
TRAINING_STEPS="${TRAINING_STEPS:-}"            # Total training steps (if set, overrides epochs)
BATCH_SIZE="${BATCH_SIZE:-8}"                   # Batch size
NUM_WORKERS="${NUM_WORKERS:-4}"                 # Dataloader workers

# ACT Policy settings
CHUNK_SIZE="${CHUNK_SIZE:-50}"                  # Action chunk size (prediction horizon)
N_ACTION_STEPS="${N_ACTION_STEPS:-50}"          # Number of action steps to execute
KL_WEIGHT="${KL_WEIGHT:-10}"                    # KL divergence weight for VAE
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}" # Vision encoder backbone

# Logging and checkpointing
LOG_FREQ="${LOG_FREQ:-100}"                     # Log every N steps
SAVE_FREQ="${SAVE_FREQ:-2000}"                  # Save checkpoint every N steps
EVAL_FREQ="${EVAL_FREQ:-0}"                     # Eval frequency (0=disabled for real robot)

# WandB settings
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-a1x_act_training}"

# Hardware settings
GPU_ID="${GPU_ID:-0}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-}"

# =============================================================================
# Parse command line arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_REPO_ID="$2"
            shift 2
            ;;
        --dataset-root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --kl-weight)
            KL_WEIGHT="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --wandb)
            WANDB_ENABLE="true"
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --log-freq)
            LOG_FREQ="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset REPO_ID      Dataset repo ID (default: a1x/yoloe_grasp_white_object)"
            echo "  --dataset-root PATH    Dataset root directory (default: ~/.cache/huggingface/lerobot)"
            echo "  --epochs N             Number of training epochs (default: 5)"
            echo "  --steps N              Training steps (overrides --epochs if set)"
            echo "  --batch-size N         Batch size (default: 8)"
            echo "  --chunk-size N         Action chunk size (default: 50)"
            echo "  --kl-weight N          KL weight for VAE (default: 10)"
            echo "  --gpu ID               GPU device ID (default: 0)"
            echo "  --output-dir PATH      Output directory (auto-generated if not set)"
            echo "  --wandb                Enable WandB logging"
            echo "  --wandb-project NAME   WandB project name (default: a1x_act_training)"
            echo "  --log-freq N           Log frequency (default: 100)"
            echo "  --save-freq N          Checkpoint save frequency (default: 2000)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup environment
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LEROBOT_DIR="$PROJECT_ROOT/refence_code/lerobot"

# Set GPU
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Generate output directory if not specified
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    TASK_NAME=$(echo "$DATASET_REPO_ID" | tr '/' '_')
    OUTPUT_DIR="$PROJECT_ROOT/outputs/train/${TIMESTAMP}_act_${TASK_NAME}"
fi

# =============================================================================
# Print configuration
# =============================================================================
echo "=============================================="
echo "ACT Training for A1X Robot"
echo "=============================================="
echo ""
echo "Dataset Configuration:"
echo "  Repo ID:        $DATASET_REPO_ID"
echo "  Root:           $DATASET_ROOT"
echo ""
echo "Training Configuration:"
echo "  Epochs:         $EPOCHS"
echo "  Steps:          $TRAINING_STEPS"
echo "  Batch Size:     $BATCH_SIZE"
echo "  Chunk Size:     $CHUNK_SIZE"
echo "  KL Weight:      $KL_WEIGHT"
echo "  Vision Backbone: $VISION_BACKBONE"
echo ""
echo "Logging Configuration:"
echo "  Log Freq:       $LOG_FREQ"
echo "  Save Freq:      $SAVE_FREQ"
echo "  WandB:          $WANDB_ENABLE"
if [ "$WANDB_ENABLE" = "true" ]; then
echo "  WandB Project:  $WANDB_PROJECT"
fi
echo ""
echo "Hardware:"
echo "  GPU:            $GPU_ID"
echo ""
echo "Output:"
echo "  Directory:      $OUTPUT_DIR"
echo ""
echo "=============================================="

# =============================================================================
# Verify environment
# =============================================================================
echo ""
echo "[1/3] Verifying environment..."

# Check if lerobot is accessible
if ! python -c "import lerobot" 2>/dev/null; then
    echo "Warning: lerobot not found in current environment."
    echo "Attempting to add lerobot to PYTHONPATH..."
    export PYTHONPATH="$LEROBOT_DIR/src:$PYTHONPATH"

    if ! python -c "import lerobot" 2>/dev/null; then
        echo "Error: Could not import lerobot. Please install it first:"
        echo "  cd $LEROBOT_DIR && pip install -e ."
        exit 1
    fi
fi
echo "  - lerobot: OK"

# Check if dataset exists
DATASET_PATH="$DATASET_ROOT/$DATASET_REPO_ID"
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset not found at $DATASET_PATH"
    echo "Please collect data first or specify correct --dataset-root"
    exit 1
fi
echo "  - Dataset: OK ($DATASET_PATH)"

# Calculate training steps from epochs if not explicitly set
if [ -z "$TRAINING_STEPS" ]; then
    # Read total_frames from dataset info.json
    INFO_JSON="$DATASET_PATH/meta/info.json"
    if [ -f "$INFO_JSON" ]; then
        TOTAL_FRAMES=$(python -c "import json; print(json.load(open('$INFO_JSON'))['total_frames'])" 2>/dev/null || echo "0")
        if [ "$TOTAL_FRAMES" -gt 0 ]; then
            # Steps per epoch = ceil(total_frames / batch_size)
            STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + BATCH_SIZE - 1) / BATCH_SIZE ))
            TRAINING_STEPS=$(( EPOCHS * STEPS_PER_EPOCH ))
            echo "  - Dataset frames: $TOTAL_FRAMES"
            echo "  - Steps per epoch: $STEPS_PER_EPOCH"
            echo "  - Total steps (${EPOCHS} epochs): $TRAINING_STEPS"
        else
            echo "Warning: Could not read total_frames from $INFO_JSON, using default 1000 steps"
            TRAINING_STEPS=1000
        fi
    else
        echo "Warning: $INFO_JSON not found, using default 1000 steps"
        TRAINING_STEPS=1000
    fi
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i $GPU_ID 2>/dev/null || echo "Not available")
    echo "  - GPU $GPU_ID: $GPU_INFO"
else
    echo "  - GPU: nvidia-smi not found, will use CPU"
fi

# =============================================================================
# Prepare output directory
# NOTE: Do NOT mkdir the output dir here. LeRobot validates that the output
#       directory does not exist (when resume=False) and will raise
#       FileExistsError if it does. Let LeRobot create it internally.
#       The train_config.txt is saved after training completes instead.
# =============================================================================
echo ""
echo "[2/3] Setting up output directory..."
echo "  Output dir: $OUTPUT_DIR (will be created by LeRobot)"

# =============================================================================
# Run training
# =============================================================================
echo ""
echo "[3/3] Starting training..."
echo "=============================================="
echo ""

# Build the training command
# Using lerobot-train CLI with draccus config system
# NOTE: --dataset.root must be the FULL path to the dataset directory
#       (i.e. $DATASET_ROOT/$DATASET_REPO_ID), not just the parent.
#       LeRobot 0.4.4 uses the root path as-is and does NOT append repo_id.
#       Passing only $DATASET_ROOT causes LeRobot to look for metadata at
#       $DATASET_ROOT/meta/info.json (wrong) instead of
#       $DATASET_ROOT/$DATASET_REPO_ID/meta/info.json (correct), which
#       triggers a fallback to HuggingFace Hub and a 404 error.
TRAIN_CMD="python -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --dataset.repo_id=$DATASET_REPO_ID \
    --dataset.root=$DATASET_ROOT/$DATASET_REPO_ID \
    --output_dir=$OUTPUT_DIR \
    --steps=$TRAINING_STEPS \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --log_freq=$LOG_FREQ \
    --save_freq=$SAVE_FREQ \
    --eval_freq=$EVAL_FREQ \
    --policy.chunk_size=$CHUNK_SIZE \
    --policy.n_action_steps=$N_ACTION_STEPS \
    --policy.kl_weight=$KL_WEIGHT \
    --policy.vision_backbone=$VISION_BACKBONE \
    --policy.push_to_hub=false"

# Add WandB if enabled
if [ "$WANDB_ENABLE" = "true" ]; then
    TRAIN_CMD="$TRAIN_CMD \
    --wandb.enable=true \
    --wandb.project=$WANDB_PROJECT"
else
    TRAIN_CMD="$TRAIN_CMD \
    --wandb.enable=false"
fi

# Print and execute
echo "Command:"
echo "$TRAIN_CMD"
echo ""

# Run training
cd "$LEROBOT_DIR"
eval $TRAIN_CMD

# =============================================================================
# Training complete — save config
# =============================================================================

# Save training configuration now that the output dir exists
cat > "$OUTPUT_DIR/train_config.txt" << EOF
# ACT Training Configuration
# Generated: $(date)

DATASET_REPO_ID=$DATASET_REPO_ID
DATASET_ROOT=$DATASET_ROOT
EPOCHS=$EPOCHS
TRAINING_STEPS=$TRAINING_STEPS
BATCH_SIZE=$BATCH_SIZE
CHUNK_SIZE=$CHUNK_SIZE
N_ACTION_STEPS=$N_ACTION_STEPS
KL_WEIGHT=$KL_WEIGHT
VISION_BACKBONE=$VISION_BACKBONE
LOG_FREQ=$LOG_FREQ
SAVE_FREQ=$SAVE_FREQ
WANDB_ENABLE=$WANDB_ENABLE
WANDB_PROJECT=$WANDB_PROJECT
GPU_ID=$GPU_ID
OUTPUT_DIR=$OUTPUT_DIR
EOF

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "To deploy this model, use:"
echo "  python imitation_learning/ACT/deploy.py --checkpoint $OUTPUT_DIR/checkpoints/last"
echo ""
