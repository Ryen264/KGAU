# KGAU

## Overview
This repository contains a knowledge graph embedding workflow focused on DirectAUKG and TransE style experiments.

Main capabilities:
- Load standard KGE datasets from `data/` (for example WN18RR, WN18, FB15k-237).
- Train and evaluate with YAML configuration files in `config/`.
- Run link prediction and triple classification evaluation.
- Save trained model checkpoints and training logs.

Main entry point:
- `main.py`

## Python Virtual Environment Setup
Create a local virtual environment in this folder if it does not exist.

```bash
cd /home/bn/KGAU
python3 -m venv .venv
```

Activate it (optional):

```bash
source .venv/bin/activate
```

You can also run commands directly with the interpreter path without activation:

```bash
./.venv/bin/python --version
```

## Install Dependencies
Install from `requirements.txt` using the local environment:

```bash
cd /home/bn/KGAU
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Run With a Config File
Run training/evaluation with a YAML config file:

```bash
cd /home/bn/KGAU
./.venv/bin/python main.py config/config_wn18rr.yaml
```

You can replace the config path with any file under `config/`, for example:
- `config/config_wn18.yaml`
- `config/config_fb15k237.yaml`

## Nohup Background Execution
This repo now includes helper scripts in `nohup/`:
- `nohup/run`
- `nohup/check`
- `nohup/stop_process`

Make scripts executable once:

```bash
cd /home/bn/KGAU
chmod +x nohup/run nohup/check nohup/stop_process
```

Start background run:

```bash
cd /home/bn/KGAU
nohup/run config/config_wn18rr.yaml
```

Check status and recent output:

```bash
cd /home/bn/KGAU
nohup/check
```

Stop background process:

```bash
cd /home/bn/KGAU
nohup/stop_process
```

Nohup helper files:
- `nohup/process.pid`: stores PID of the current background run.
- `nohup/nohup.out`: stdout/stderr from the background process.

## Logs and Output Files
Training logs:
- Main script writes timestamped logs under `logs/<dataset>/comparison/`.
- Example filename pattern: `compare_directaukg_gamma_sweep_YYMMDD-HHMMSS.log`.

Model checkpoints:
- Saved under `models/<dataset>/<task>/components/`.

Output folder:
- `output/` is available as a general output directory.
- Current training flow primarily uses `logs/` and `models/`; if you add exports/reports, `output/` is a good target location.

## Quick Start
```bash
cd /home/bn/KGAU
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python main.py config/config_wn18rr.yaml
```
