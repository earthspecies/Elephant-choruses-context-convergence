# Elephant Rumble Context Prediction

Classifies context type of elephant rumbles using acoustic features and logistic regression. Supports within-dataset cross-validation and cross-dataset generalization experiments (i.e. training and testing on same dataset as well as training and testing on different datasets).

## Files

| File | Description |
|------|-------------|
| `predict_context.py` | Main script: feature extraction, model training, evaluation |
| `utils.py` | Shared utilities: metadata filtering, confusion matrix plotting |

## Data structure

```
rumbles/
├── EB/
│   ├── single/          # wav files + cached features
│   ├── single_metadata.csv
│   ├── chorus/
│   └── chorus_metadata.csv
└── nonEB/
    ├── single/
    ├── single_metadata.csv
    ├── chorus/
    └── chorus_metadata.csv
```

Metadata CSVs require columns: `SndFile`, `ContextTypeNew`, `CallerName` (or `Caller`), `AgeCaller`/`AgeClassCaller`.

Calves (age code `0B`) are excluded by default (`EXCLUDE_CALVES = True`). Context classes with fewer than `SMALL_CLASS_THRESH` (default: 5) samples are dropped.

## Features

Two feature types are extracted per call and aggregated to the conversation level:

- **`loglinpow-30-80`** — 30 log-linear power bands (80 Hz–8 kHz), aggregated by mean across calls (note all audio files are pitch shifted)
- **`acoustic-contour-feats`** — mel spectrogram PCA (10 components), aggregated by mean and std across calls

Features are cached in "rumbles" directory:
- Loglinpow: `rumbles/{dataset}/{subset}/{filename}/{feature}.npy`
- Mel spectrogram: `rumbles/{dataset}/{subset}/{filename}_powerbands.npy`

Set `RECOMPUTE = False` (default) to load from cache. Set `RECOMPUTE = True` to recompute from wav files.

## Cross-validation

- **Within-dataset** (no `--dataset-test`): 20-fold cross-validation (`N_FOLDS = 20`)
- **With `--test-on-held-out-callers`**: Leave-One-Caller-Out (LOGO) cross-validation
- **Cross-dataset** (`--dataset-test`): train on full training set, test on held-out dataset

## Usage

```bash
# Within-dataset, k-fold
python3.11 predict_context.py --dataset EB --subset single --exp-name EB_single

# Within-dataset, leave-one-caller-out
python3.11 predict_context.py --dataset EB --subset single --test-on-held-out-callers --exp-name EB_single-loo

# Cross-dataset (train on nonEB single, test on EB single)
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --dataset-test EB --subset-test single \
    --exp-name EB_single_train_nonEB_single

# see additional examples in python files
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--dataset` | Yes | Training data: `EB` or `nonEB` |
| `--subset` | Yes | Call type: `single` or `chorus` |
| `--dataset-test` | No | Test dataset (defaults to `--dataset`) |
| `--subset-test` | No | Test call type (defaults to `--subset`) |
| `--test-on-held-out-callers` | No | Use LOGO cross-validation |
| `--exp-name` | No | Name for output folder (auto-generated if omitted) |

## Output

Results are written to `context_prediction_output/{exp-name}/`:

| File | Description |
|------|-------------|
| `{exp-name}_results.txt` | CSV with accuracy, F1, baselines, p-value |
| `{exp-name}_classif_report.txt` | Per-class precision/recall/F1 |
| `{exp-name}_conf_mat.png` | Confusion matrix |
| `{exp-name}_predictions.csv` | Per-sample predictions |
| `{exp-name}-top-importances.txt` | Top feature importances |
| `significance_info.json` | Fold indices and raw predictions |

### Results columns

`experiment_name`, `train_data`, `test_data`, `n_samples`, `n_classes`, `baseline_prop`, `baseline_majority`, `pred_dist_mean`, `f1_macro`, `accuracy`, `p_value_pred_distribution`

The p-value is from a permutation test corresponding to the fraction of times shuffled predictions match ground truth at least as well as actual predictions.

### Summary table

```bash
python3.11 summarize_results.py
# prints table and saves to context_prediction_output/summary.txt
```


## Dependencies

```
numpy pandas librosa scikit-learn scipy seaborn matplotlib natsort
```

