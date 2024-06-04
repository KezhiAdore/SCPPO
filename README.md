# SCPPO: Surefirely Constrained Proximal Policy Optimization

## Introduction

This repo is the official implementation of SCPPO

## Run

1. Create a virtual environment using `conda` and activate it:

```bash
conda create -n scppo python=3.10 -y
conda activate scppo
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Run the training script:

```bash
# run scppo 
python run.py --env dm_control/ball_in_cup-catch-v0 --device cpu --seed 0
# run stable baselines3
python sb_run.py --env dm_control/ball_in_cup-catch-v0 --algo ppo --device cpu --seed 0
```

4. Use `tensorboard` to visualize the training process:

```bash
tensorboard --logdir logs
```