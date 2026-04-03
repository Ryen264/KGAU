# KGAU

Training and evaluation code for comparing `DirectAUKG` and `TransE` on knowledge graph benchmarks (default: WN18RR).

## Requirements

- Linux (tested in this workspace)
- Python 3.10+ (or compatible version in your virtual environment)
- Packages from `requirements.txt`

Install dependencies:

```bash
/home/bn/GBPU-KBGAN/.venv/bin/python -m pip install -r /home/bn/KGAU/requirements.txt
```

## Dataset Layout

The project expects datasets under `data/`.

For WN18RR, required files are:

- `data/wn18rr/train.txt`
- `data/wn18rr/valid.txt`
- `data/wn18rr/test.txt`
- `data/wn18rr_w_labels/valid.txt`
- `data/wn18rr_w_labels/test.txt`

## Run

From anywhere, using your current virtual environment Python:

```bash
/home/bn/GBPU-KBGAN/.venv/bin/python /home/bn/KGAU/main.py config/config_wn18rr.yaml
```

Equivalent command from the project root:

```bash
cd /home/bn/KGAU
/home/bn/GBPU-KBGAN/.venv/bin/python main.py config/config_wn18rr.yaml
```

## Optional Arguments

You can override common settings from CLI:

```bash
/home/bn/GBPU-KBGAN/.venv/bin/python /home/bn/KGAU/main.py config/config_wn18rr.yaml \
  --gpu 0 \
  --seed 42 \
  --direct_n_epoch 200 \
  --transe_n_epoch 200
```

Useful flags:

- `--no_log_to_file`: Disable file logging
- `--early_stop_patience N`: Enable early stopping (`-1` disables it)
- `--test_batch_size N`: Evaluation batch size

## Output

- Console summary comparing both models
- Log files under `logs/<dataset>/comparison/` (unless `--no_log_to_file` is set)
