# Acoustic Convergence in Elephant Rumbles

Analysis of whether elephant vocal exchanges show acoustic convergence (becoming more similar over the course of an exchange) across different communicative contexts.

## Overview

This pipeline extracts log-power spectral features from elephant rumbles organized by exchange, then measures whether successive rumbles become acoustically more similar over time. Significance is determined using permutation tests that shuffle either the feature order within rumble positions or features across all rumbles globally.

## Directory Structure

```
convergence/
├── convergence_rumbles.py       # Main analysis script
├── train.py                     # Acoustic feature extraction utilities
├── utils.py                     # Helper functions (metadata filtering, result saving)
├── conv_data_EB.csv             # Rumble metadata (caller, context, file paths)
├── caller_summary.csv           # Per-conversation rumble counts and context types
├── rumble_features_*.csv        # Pre-computed acoustic features (519 rumbles)
├── sounds_filt/                 # Audio and annotation files, organized by conversation
│   └── {ID}_{ContextType}/
│       ├── selections.txt       # Raven annotations (rumble timing, caller ID)
│       └── *.wav                # Filtered audio files
└── output/                      # Results: CSVs and plots
```

## To run analysis code

```bash
# With pre-computed features (default):
python convergence_rumbles.py

```

Output is written to `output/`:
- `convergence_by_conversation_*.csv` — per-conversation convergence and adjacency distances
- `convergence_by_context_*.csv` — per-context aggregated convergence with p-values
- `convergence_barplot_*.png` — bar chart of mean convergence by context
- `convergence_permhist_*.png` — permutation test null distributions



## Details about files included here and analysis steps

## Conversation Folders

Each subfolder in `sounds_filt/` is named `{ConversationID}_{ContextType}` (e.g., `247_Greeting`, `103_Contact`). Context types include:

- **Greeting** — close-range social greetings
- **Contact** — long-distance contact calls
- **Cadenced** — rhythmically structured rumbles
- **Little greeting** — brief affiliative exchanges
- **Coo** — softer, contact-type calls
- **Lets go** — movement-coordination rumbles

Each folder contains a `selections.txt` annotation file (Raven format) with columns: `Selection`, `View`, `Channel`, `Begin Time (s)`, `End Time (s)`, `Low Freq (Hz)`, `High Freq (Hz)`.

## Feature Extraction

Features are computed by `compute_log_power_in_bands()` in [train.py](train.py):

- STFT with `n_fft=800`, `hop_length=128`
- Spectrum divided into 30 bands × 80 Hz = 2400 Hz total range (80–2400 Hz)
- Log-power averaged within each band; lowest band excluded → **29 features per rumble**
- Feature set label: `loglinpow-30-80`

## Convergence Metric

For each exchange, convergence is computed as the negative slope of a linear regression fit to `log(Euclidean distance in feature space)` between adjacent rumbles vs. rumble sequence position. A positive convergence value indicates convergence; negative indicates divergence.

2 permutation tests evaluate significance:
1. **Order-shuffle**: shuffles feature vectors within each rumble position across conversations, preserving per-position variance
2. **Global shuffle**: shuffles features across all rumbles in all conversations


## Dependencies

- Python 3.x
- `numpy`, `scipy`, `pandas`, `matplotlib`
- `librosa` (audio loading and STFT)
- `sklearn` (linear regression)
