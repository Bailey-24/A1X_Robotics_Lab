# ACT Training for A1X Robot

This directory contains training scripts for the **ACT (Action Chunking Transformers)** policy using the [LeRobot](https://github.com/huggingface/lerobot) framework.

## Overview

ACT is an imitation learning policy that predicts a sequence ("chunk") of future actions from observations. It uses a VAE-based architecture with:
- **Vision encoder**: ResNet18 for processing camera images
- **Transformer encoder**: For encoding observations
- **Transformer decoder**: For decoding action sequences
- **VAE**: For learning a latent style variable during training

## Prerequisites

### 1. Environment Setup

```bash
conda activate a1x_ros
cd /home/ubuntu/projects/A1Xsdk
```

### 2. Install LeRobot

```bash
cd refence_code/lerobot
pip install -e .
```

### 3. Collect Training Data

Use the data collection pipeline to collect demonstrations:
```bash
# See data_collection/README.md for detailed instructions
python data_collection/collect_demos.py --task-name my_task
```

The collected data should be converted to LeRobot format and stored at:
```
~/.cache/huggingface/lerobot/a1x/<task_name>/
```

## Training

### Quick Start (5 Epochs on Test Dataset)

```bash
# Train with default settings (5 epochs on yoloe_grasp_white_object test dataset)
cd /home/ubuntu/projects/A1Xsdk
./imitation_learning/ACT/train.bash

# The script will automatically calculate steps:
# - Test dataset: 534 frames, batch_size=8 → 67 steps/epoch
# - 5 epochs → 335 total training steps
```

### Custom Training

```bash
# Train on a custom dataset for 10 epochs
./imitation_learning/ACT/train.bash --dataset a1x/my_task_name --epochs 10

# Train for specific number of steps (overrides epochs)
./imitation_learning/ACT/train.bash --steps 50000

# Enable WandB logging
./imitation_learning/ACT/train.bash --wandb --wandb-project my_project
```

### Full Options

```bash
./train.bash [OPTIONS]

Options:
  --dataset REPO_ID      Dataset repo ID (default: a1x/yoloe_grasp_white_object)
  --dataset-root PATH    Dataset root directory (default: ~/.cache/huggingface/lerobot)
  --epochs N             Number of training epochs (default: 5)
  --steps N              Training steps (overrides --epochs if set)
  --batch-size N         Batch size (default: 8)
  --chunk-size N         Action chunk size (default: 50)
  --kl-weight N          KL weight for VAE (default: 10)
  --gpu ID               GPU device ID (default: 0)
  --output-dir PATH      Output directory (auto-generated if not set)
  --wandb                Enable WandB logging
  --wandb-project NAME   WandB project name (default: a1x_act_training)
  --log-freq N           Log frequency (default: 100)
  --save-freq N          Checkpoint save frequency (default: 2000)
  -h, --help             Show this help message
```

### Training Tips

1. **Data quantity**: ACT typically needs **50-100+ episodes** for good performance. With only 1 episode (test data), expect poor results.

2. **Epochs vs Steps**:
   - Use `--epochs N` for epoch-based training (recommended, auto-calculates steps)
   - Use `--steps N` to override with exact step count
   - Steps per epoch = ceil(total_frames / batch_size)

3. **Chunk size**: Set based on your task duration:
   - Fast tasks (< 3 seconds): `--chunk-size 30`
   - Normal tasks (3-5 seconds): `--chunk-size 50`
   - Slow tasks (> 5 seconds): `--chunk-size 100`

4. **KL weight**: Controls VAE regularization:
   - Default: `--kl-weight 10`
   - For small datasets, try increasing: `--kl-weight 50-100`

5. **Training epochs**: Rule of thumb:
   - Quick test/validation: 5 epochs
   - Small dataset (< 20 episodes): 50-100 epochs
   - Medium dataset (20-50 episodes): 100-300 epochs
   - Large dataset (50+ episodes): 300-1000 epochs

## Test Dataset

A test dataset is available for verifying the training workflow:

```
~/.cache/huggingface/lerobot/a1x/yoloe_grasp_white_object/
```

**Dataset Info**:
- **Robot**: A1X (6-DOF arm + gripper)
- **Episodes**: 1
- **Frames**: 534
- **FPS**: 20
- **Features**:
  - `observation.state`: [7] - 6 joint positions + gripper
  - `action`: [7] - 6 joint actions + gripper
  - `observation.images.cam_wrist`: [3, 480, 640] - Wrist camera RGB

**Test Run** (5 epochs, ~335 steps):
```bash
./imitation_learning/ACT/train.bash
# Takes ~6 minutes on RTX 4080 GPU
```

## Output Structure

After training, the output directory contains:

```
outputs/train/YYYY-MM-DD_HH-MM-SS_act_a1x_task/
├── train_config.txt           # Training configuration
├── checkpoints/
│   ├── 002000/                # Checkpoint at step 2000
│   │   ├── config.json        # Policy configuration
│   │   ├── model.safetensors  # Model weights
│   │   ├── preprocessor.pt    # Input preprocessor
│   │   └── postprocessor.pt   # Output postprocessor
│   ├── 004000/                # Checkpoint at step 4000
│   └── last -> 010000/        # Symlink to latest checkpoint
└── logs/                      # Training logs
```

## Deployment

After training, deploy the model on the real A1X robot:

```bash
# Deploy the trained policy (to be implemented)
python imitation_learning/ACT/deploy.py --checkpoint outputs/train/.../checkpoints/last
```

See `deploy.py` for the deployment script (to be created after training verification).

## Troubleshooting

### "Output directory already exists and resume is False"
LeRobot validates that the output directory does not exist when `resume=False`. If the output directory is pre-created before training (e.g. by `mkdir -p`), training will fail with `FileExistsError`. The training script lets LeRobot create the output directory internally. If you see this error, delete the stale output directory and re-run:
```bash
rm -rf outputs/train/<the_failed_run_dir>
./imitation_learning/ACT/train.bash
```

### "Repository Not Found" / 404 on HuggingFace Hub (local dataset)
This happens when `--dataset.root` points to the parent cache directory (e.g. `~/.cache/huggingface/lerobot`) instead of the full dataset path. LeRobot 0.4.4 uses `root` as-is without appending `repo_id`, so it fails to find local metadata and falls back to the Hub. The training script already handles this by passing `$DATASET_ROOT/$DATASET_REPO_ID` as `--dataset.root`. If you see this error when running manually, make sure root is the full path:
```bash
# Wrong — LeRobot looks for metadata at ~/.cache/huggingface/lerobot/meta/info.json
--dataset.root=~/.cache/huggingface/lerobot

# Correct — full path to the dataset directory
--dataset.root=~/.cache/huggingface/lerobot/a1x/yoloe_grasp_white_object
```

### "lerobot not found"
```bash
cd /home/ubuntu/projects/A1Xsdk/refence_code/lerobot
pip install -e .
```

### "Dataset not found"
Ensure your dataset is at the correct path:
```bash
ls ~/.cache/huggingface/lerobot/a1x/<your_task_name>/
```

### Out of GPU memory
Reduce batch size:
```bash
./train.bash --batch-size 4
```

### Training loss not decreasing
- Check if you have enough data (50+ episodes recommended)
- Try increasing KL weight: `--kl-weight 50`
- Verify data quality by visualizing: `python -m lerobot.scripts.visualize_dataset --repo-id a1x/your_task`




# eval Custom checkpoint
python imitation_learning/ACT/deploy.py --checkpoint outputs/train/.../checkpoints/last/pretrained_model


## References

- [ACT Paper](https://arxiv.org/abs/2304.13705): Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [LeRobot ACT Policy](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/act)
