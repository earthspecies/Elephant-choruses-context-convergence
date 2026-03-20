import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import os
import glob
import sys
import hashlib
import librosa
from scipy.signal import butter, filtfilt

# Make seed deterministic using hash
def stable_seed(s):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)

# Import feature extraction functions from train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import compute_log_power_in_bands

# Hard-coded parameters
TRUNCATE_RUMBLES = 10
FILTER_CONTEXT = None
MIN_RUMBLES = 3

metadata_file = Path(__file__).parent / 'conv_data_EB.csv'

FEAT = 'loglinpow-30-80'
ALLOW_OVERLAPS = False
LOAD_SAVED_FEATURES = True  # Set to True to skip WAV processing and load saved features CSV




# Only include folders listed in caller_summary_ANNOT.csv (all rows meet caller criteria)
_CALLER_SUMMARY = Path(__file__).parent / 'caller_summary.csv'
if _CALLER_SUMMARY.exists():
    _cs = pd.read_csv(_CALLER_SUMMARY)
    CALLER_CRITERIA_FOLDERS = set(_cs['folder'].tolist())
    print(f"Found {len(CALLER_CRITERIA_FOLDERS)} folders meeting caller criteria")
else:
    CALLER_CRITERIA_FOLDERS = None
    print("WARNING: caller_summary_ANNOT.csv not found, no caller-based filtering applied")

EXCLUDE_FOLDERS = []

# Summary statistic for aggregating convergence across conversations
agg_func = np.mean
agg_name = 'mean'

# Seed controlling all permutation shuffles. Change this number to get different permutations.
PERM_SEED = 100

# Create a simple args object for train.py functions
class SimpleArgs:
    def __init__(self, ignore_first_band):
        self.ignore_first_band = ignore_first_band

ignore_first_band = True
ARGS = SimpleArgs(ignore_first_band)

# Set up paths
sounds_dir = Path(__file__).parent / 'sounds_filt'
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)


print("=" * 80)
print("CONVERGENCE ANALYSIS WITH RUMBLE-LEVEL SEGMENTATION")
print("=" * 80)
print(f"Sounds directory: {sounds_dir}")
print(f"Output directory: {output_dir}\n")

# Parse feature configuration
feat_parts = FEAT.split('-')
n_bands = int(feat_parts[1])
bandwidth = int(feat_parts[2])
max_freq = n_bands * bandwidth

print(f"Using {n_bands} bands with {bandwidth} Hz bandwidth each")
print(f"Frequency range: 0-{max_freq} Hz")
if max_freq > 2400:
    print(f"WARNING: Max frequency {max_freq} Hz exceeds recommended 2400 Hz")

if ignore_first_band:
    print(f"Ignoring lowest band (0-{bandwidth} Hz)")
    feature_start_idx = 1
    feature_names = [f'logpow_band{i}' for i in range(1, n_bands)]
else:
    feature_start_idx = 0
    feature_names = [f'logpow_band{i}' for i in range(n_bands)]

# print(f"Using features: {feature_names[:3]}...{feature_names[-3:]}\n")


def bandpass_filter(audio, sr, lowcut=80, highcut=2400):
    """Apply bandpass filter to audio"""
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')
    filtered = filtfilt(b, a, audio)
    return filtered


def load_selection_table(folder_path):
    """
    Read selections.txt and return a list of (begin_time, end_time) tuples,
    one per unique selection, in selection-number order.
    """
    sel_file = folder_path / 'selections.txt'
    if not sel_file.exists():
        return None

    df = pd.read_csv(sel_file, sep='\t')

    # Keep only Waveform rows to get one row per selection
    wf = df[df['View'] == 'Waveform 1'].copy()
    wf = wf.sort_values('Begin Time (s)').reset_index(drop=True)

    rumbles = []
    for _, row in wf.iterrows():
        rumbles.append({
            'selection': int(row['Selection']),
            'begin': float(row['Begin Time (s)']),
            'end': float(row['End Time (s)'])
        })

    return rumbles


def resolve_overlaps(rumbles):
    """
    For overlapping rumbles, trim both start and end times to avoid overlaps.
    This matches the behavior in split_convs.py/utils.py:
    - Trim each rumble's start to the previous rumble's original end
    - Trim each rumble's end to the next rumble's original start
    - Last rumble uses original times (no trimming)
    Returns a new list with adjusted times, skipping any that become zero-length.
    """
    if len(rumbles) <= 1:
        return rumbles

    n = len(rumbles)
    resolved = []
    prev_original_end = 0

    for i in range(n):
        r = rumbles[i].copy()

        # Trim start: can't start before previous rumble's original end
        split_start = max(prev_original_end, r['begin'])

        if i == n - 1:
            # Last segment: only trim start, keep original end
            split_end = r['end']
        else:
            # Trim end: can't extend past next rumble's original start
            next_start = rumbles[i + 1]['begin']
            split_end = min(r['end'], next_start)

        # Skip if zero or negative duration
        if split_end <= split_start:
            print(f"  Skipping selection {r['selection']}: complete overlap")
            continue  # Don't update prev_original_end (matches utils.py)

        r['begin'] = split_start
        r['end'] = split_end
        resolved.append(r)
        prev_original_end = rumbles[i]['end']  # Use ORIGINAL end, not trimmed

    return resolved


def process_conversation_folder(folder_path):
    """
    Process one conversation folder:
    1. Read selection table for rumble times
    2. Load and concatenate audio files
    3. Extract rumble segments based on selection times
    4. Extract features for each rumble
    """
    folder_name = folder_path.name

    # Parse folder name: {random#}_{context}
    parts = folder_name.rsplit('_', 1)
    if len(parts) != 2:
        return {
            'status': 'skipped',
            'reason': 'bad_folder_name',
            'folder_name': folder_name,
            'n_original': 0, 'n_after_overlap': 0, 'n_used': 0
        }

    conv_id = folder_name
    context_type = parts[1]

    # Load selection table
    rumbles = load_selection_table(folder_path)
    if rumbles is None or len(rumbles) == 0:
        return {
            'status': 'skipped',
            'reason': 'no_selections',
            'folder_name': folder_name,
            'n_original': 0, 'n_after_overlap': 0, 'n_used': 0
        }

    # Find all wav files in folder
    audio_files = sorted(glob.glob(str(folder_path / "*.wav")))
    audio_files += sorted(glob.glob(str(folder_path / "*.WAV")))

    if len(audio_files) == 0:
        return {
            'status': 'skipped',
            'reason': 'no_audio',
            'folder_name': folder_name,
            'n_original': len(rumbles), 'n_after_overlap': 0, 'n_used': 0
        }

    # Load and concatenate all audio files
    concatenated_audio = []
    sample_rate = None

    for audio_file in sorted(audio_files):
        try:
            audio, sr = librosa.load(audio_file, sr=None)

            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

            concatenated_audio.append(audio)
        except Exception as e:
            print(f"  Error loading {audio_file}: {e}")
            continue

    if len(concatenated_audio) == 0:
        return {
            'status': 'skipped',
            'reason': 'audio_load_failed',
            'folder_name': folder_name,
            'n_original': len(rumbles), 'n_after_overlap': 0, 'n_used': 0
        }

    full_audio = np.concatenate(concatenated_audio)

    # Apply bandpass filter
    filtered_audio = bandpass_filter(full_audio, sample_rate, lowcut=100, highcut=6000)

    # Resolve overlaps if requested
    if ALLOW_OVERLAPS:
        segments = rumbles
    else:
        segments = resolve_overlaps(rumbles)

    # Extract features for each rumble segment
    rumble_features = []

    for seg in segments:
        start_sample = int(seg['begin'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)

        # Clip to audio bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(filtered_audio), end_sample)

        if end_sample <= start_sample:
            continue

        segment_audio = filtered_audio[start_sample:end_sample]

        # compute_log_power_in_bands needs at least n_fft=800 samples
        if len(segment_audio) < 800:
            continue

        try:
            features = compute_log_power_in_bands(segment_audio, n_bands, bandwidth,
                                                  sr=sample_rate, ARGS=ARGS)
            if features is not None:
                features_to_use = features[feature_start_idx:]
                rumble_features.append(features_to_use)
        except Exception as e:
            print(f"  Error extracting features for selection {seg['selection']} in {folder_name}: {e}")
            continue

    n_original = len(rumbles)
    n_after_overlap = len(segments)
    n_used = len(rumble_features)

    if n_used < MIN_RUMBLES:
        return {
            'status': 'skipped',
            'reason': 'too_few_rumbles',
            'folder_name': folder_name,
            'n_original': n_original,
            'n_after_overlap': n_after_overlap,
            'n_used': n_used
        }

    return {
        'status': 'success',
        'conversation_id': conv_id,
        'context_type': context_type,
        'features': np.array(rumble_features),
        'n_rumbles': n_used,
        'n_original': n_original,
        'n_after_overlap': n_after_overlap
    }


# Output file tags (used for both saving and loading)
overlap_tag = 'overlaps' if ALLOW_OVERLAPS else 'nooverlaps'
median_tag = ''
truncate_tag = f'_last{TRUNCATE_RUMBLES}' if TRUNCATE_RUMBLES is not None else ''
features_output_file = Path(__file__).parent / f'rumble_features_{FEAT}_{overlap_tag}{truncate_tag}{median_tag}.csv'

if LOAD_SAVED_FEATURES:
    print("=" * 80)
    print("LOADING SAVED FEATURES")
    print("=" * 80)
    if not features_output_file.exists():
        print(f"ERROR: saved features file not found: {features_output_file}")
        sys.exit(1)
    df = pd.read_csv(features_output_file)
    feature_names = [c for c in df.columns if c.startswith('logpow_band')]
    print(f"Loaded {len(df)} rumbles from {df['conversation_id'].nunique()} conversations\n")
else:
    # Load metadata to filter folders
    print("=" * 80)
    print("LOADING METADATA")
    print("=" * 80)

    metadata_df = pd.read_csv(metadata_file)
    print(f"Loaded metadata with {len(metadata_df)} rows")

    # Get unique context groups (folder names) from metadata
    allowed_folders = set(metadata_df['context_group'].dropna().unique())
    print(f"Found {len(allowed_folders)} unique folders in metadata\n")

    # Process all conversation folders
    print("=" * 80)
    print("PROCESSING CONVERSATION FOLDERS")
    print("=" * 80)

    all_folders = [f for f in sounds_dir.iterdir() if f.is_dir()]
    print(f"Found {len(all_folders)} folders in directory")

    conversation_folders = [f for f in all_folders
                            if f.name in allowed_folders
                            and f.name not in EXCLUDE_FOLDERS
                            and (CALLER_CRITERIA_FOLDERS is None or f.name in CALLER_CRITERIA_FOLDERS)]
    print(f"Filtered to {len(conversation_folders)} folders from metadata")
    print(f"Excluded {len(all_folders) - len(conversation_folders)} folders not in metadata\n")

    all_conversations = []
    skipped_results = []
    exclusion_reasons = {}

    for folder in sorted(conversation_folders):
        result = process_conversation_folder(folder)
        if result['status'] == 'success':
            all_conversations.append(result)
        else:
            skipped_results.append(result)
            reason = result['reason']
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + 1

    # Print detailed exclusion summary
    print(f"\n{'='*60}")
    print("FOLDER PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total folders processed: {len(conversation_folders)}")
    print(f"Successfully included:   {len(all_conversations)}")
    print(f"Excluded:                {len(skipped_results)}")

    if exclusion_reasons:
        print(f"\nExclusion reasons:")
        reason_labels = {
            'bad_folder_name': 'Bad folder name format',
            'no_selections': 'No selection table or empty',
            'no_audio': 'No audio files found',
            'audio_load_failed': 'Audio failed to load',
            'too_few_rumbles': 'Fewer than 3 valid rumbles'
        }
        for reason, count in sorted(exclusion_reasons.items(), key=lambda x: -x[1]):
            label = reason_labels.get(reason, reason)
            print(f"  - {label}: {count}")

    # Print per-folder details for excluded folders
    if skipped_results:
        print(f"\nExcluded folders detail:")
        for r in skipped_results:
            print(f"  {r['folder_name']}: {r['reason']} "
                  f"(orig={r['n_original']}, after_overlap={r['n_after_overlap']}, used={r['n_used']})")

    # Print selection statistics for included folders
    print(f"\n{'='*60}")
    print("SELECTION STATISTICS (included folders)")
    print(f"{'='*60}")
    total_original = sum(c['n_original'] for c in all_conversations)
    total_after_overlap = sum(c['n_after_overlap'] for c in all_conversations)
    total_used = sum(c['n_rumbles'] for c in all_conversations)

    print(f"Total original selections:      {total_original}")
    print(f"After overlap resolution:       {total_after_overlap} ({100*total_after_overlap/total_original:.1f}%)" if total_original > 0 else "")
    print(f"Used in analysis (>=800 samp):  {total_used} ({100*total_used/total_original:.1f}%)" if total_original > 0 else "")
    print(f"Selections lost to overlaps:    {total_original - total_after_overlap}")
    print(f"Selections lost to short audio: {total_after_overlap - total_used}")

    # Rumbles per conversation statistics
    rumbles_per_conv = np.array([c['n_rumbles'] for c in all_conversations])
    print(f"\nRumbles per conversation (used):")
    print(f"  Mean: {np.mean(rumbles_per_conv):.2f}")
    print(f"  SE:   {np.std(rumbles_per_conv) / np.sqrt(len(rumbles_per_conv)):.2f}")
    print(f"  Range: {np.min(rumbles_per_conv)} - {np.max(rumbles_per_conv)}")

    # Conversations per context statistics
    context_counts = {}
    for c in all_conversations:
        ct = c['context_type']
        context_counts[ct] = context_counts.get(ct, 0) + 1

    print(f"\n{'='*60}")
    print("CONVERSATIONS PER CONTEXT")
    print(f"{'='*60}")
    print(f"{'Context':<20} {'N convs':>8}")
    print("-" * 30)
    for ct in sorted(context_counts.keys()):
        print(f"{ct:<20} {context_counts[ct]:>8}")

    convs_per_context = np.array(list(context_counts.values()))
    print(f"\nSummary across {len(context_counts)} context types:")
    print(f"  Mean: {np.mean(convs_per_context):.2f} conversations/context")
    print(f"  SE:   {np.std(convs_per_context) / np.sqrt(len(convs_per_context)):.2f}")
    print(f"  Range: {np.min(convs_per_context)} - {np.max(convs_per_context)}")

    # Per-conversation breakdown
    print(f"\nPer-conversation breakdown:")
    print(f"{'Folder':<40} {'Original':>8} {'AfterOvlp':>10} {'Used':>6}")
    print("-" * 66)
    for c in sorted(all_conversations, key=lambda x: x['conversation_id']):
        print(f"{c['conversation_id']:<40} {c['n_original']:>8} {c['n_after_overlap']:>10} {c['n_rumbles']:>6}")

    if len(all_conversations) == 0:
        print("ERROR: No conversations processed successfully!")
        sys.exit(1)

    # Create DataFrame with all rumbles
    print("=" * 80)
    print("CREATING DATASET")
    print("=" * 80)

    all_data = []
    for conv in all_conversations:
        for rumble_idx, features in enumerate(conv['features']):
            record = {
                'conversation_id': conv['conversation_id'],
                'context_type': conv['context_type'],
                'rumble_number': rumble_idx + 1
            }
            for feat_name, feat_val in zip(feature_names, features):
                record[feat_name] = feat_val
            all_data.append(record)

    df = pd.DataFrame(all_data)
    print(f"Created dataset with {len(df)} rumbles from {df['conversation_id'].nunique()} conversations")


    # Truncate to last N rumbles per conversation (hard-coded)
    if TRUNCATE_RUMBLES is not None:
        print(f"\n{'='*60}")
        print(f"TRUNCATING TO LAST {TRUNCATE_RUMBLES} RUMBLES PER CONVERSATION")
        print(f"{'='*60}")

        n_before = len(df)
        n_convs_before = df['conversation_id'].nunique()

        # For each conversation, keep only the last N rumbles
        truncated_dfs = []
        n_truncated = 0
        for conv_id, group in df.groupby('conversation_id'):
            if len(group) > TRUNCATE_RUMBLES:
                # Keep last N rumbles (highest rumble_numbers)
                kept = group.nlargest(TRUNCATE_RUMBLES, 'rumble_number').sort_values('rumble_number')
                # Renumber from 1
                kept = kept.copy()
                kept['rumble_number'] = range(1, len(kept) + 1)
                truncated_dfs.append(kept)
                n_truncated += 1
            else:
                truncated_dfs.append(group)

        df = pd.concat(truncated_dfs, ignore_index=True)
        n_after = len(df)

        print(f"\n  Conversations truncated: {n_truncated}/{n_convs_before}")
        print(f"  Rumbles before: {n_before}")
        print(f"  Rumbles after:  {n_after} (removed {n_before - n_after})")
        print(f"  All {df['conversation_id'].nunique()} conversations retained")
        print(f"  Range: {df.groupby('conversation_id').size().min()}-{df.groupby('conversation_id').size().max()} rumbles/conversation")

        if n_truncated > 0:
            print(f"\n  Truncated conversations:")
            trunc_df = df.groupby(['context_type', 'conversation_id']).size().reset_index(name='n_now')
            for ct in sorted(trunc_df['context_type'].unique()):
                ct_data = trunc_df[trunc_df['context_type'] == ct]
                ct_trunc = ct_data[ct_data['n_now'] == TRUNCATE_RUMBLES]
                if len(ct_trunc) > 0:
                    print(f"    {ct}: {len(ct_trunc)} conversations truncated to {TRUNCATE_RUMBLES}")

    # Show statistics by context type
    print("\nRumbles per conversation by context type:")
    print(f"{'Context':<20} {'N convs':>8} {'Mean':>8} {'SE':>8} {'Range':>12}")
    print("-" * 60)
    for context_type in sorted(df['context_type'].unique()):
        context_df = df[df['context_type'] == context_type]
        n_convs = context_df['conversation_id'].nunique()
        rumbles_per_conv = context_df.groupby('conversation_id').size()
        mean_r = rumbles_per_conv.mean()
        se_r = rumbles_per_conv.std() / np.sqrt(n_convs) if n_convs > 1 else 0
        range_str = f"{rumbles_per_conv.min()}-{rumbles_per_conv.max()}"
        print(f"{context_type:<20} {n_convs:>8} {mean_r:>8.1f} {se_r:>8.2f} {range_str:>12}")

    # Overall rumbles per conversation (post-filtering/truncation)
    all_rumbles_per_conv = df.groupby('conversation_id').size()
    overall_mean = all_rumbles_per_conv.mean()
    overall_se = all_rumbles_per_conv.std() / np.sqrt(len(all_rumbles_per_conv))
    print("-" * 60)
    print(f"{'OVERALL':<20} {len(all_rumbles_per_conv):>8} {overall_mean:>8.1f} {overall_se:>8.2f} {f'{all_rumbles_per_conv.min()}-{all_rumbles_per_conv.max()}':>12}")

    # Save extracted features
    df.to_csv(features_output_file, index=False, float_format='%.2f')
    print(f"\nSaved features to: {features_output_file}\n")



# ============================================================================
# CONVERGENCE CALCULATION
# ============================================================================

def calculate_convergence(data, feature_col):
    """Calculate convergence rate for a single feature"""
    if len(data) < 3:
        return np.nan

    values = data[feature_col].values
    successive_diffs = np.abs(np.diff(values))
    successive_diffs = successive_diffs[successive_diffs > 0]

    if len(successive_diffs) < 2:
        return np.nan

    log_diffs = np.log(successive_diffs)
    sequence_idx = np.arange(len(log_diffs))

    slope, _, _, _, _ = stats.linregress(sequence_idx, log_diffs)
    slope *= len(successive_diffs)

    return -slope


def calculate_overall_convergence(data, feature_cols):
    """Calculate overall convergence using Euclidean distances"""
    if len(data) < 3:
        return np.nan

    feature_matrix = data[feature_cols].values
    successive_distances = np.sqrt(np.sum(np.diff(feature_matrix, axis=0)**2, axis=1))
    successive_distances = successive_distances[successive_distances > 0]

    if len(successive_distances) < 2:
        return np.nan

    log_distances = np.log(successive_distances)
    sequence_idx = np.arange(len(log_distances))
    slope, _, _, _, _ = stats.linregress(sequence_idx, log_distances)
    slope *= len(successive_distances)

    return -slope


def gt_no_ties(scores, true_val):
    """Fraction of scores strictly greater than true_val, excluding ties."""
    scores = np.array(scores)
    n_gthan = np.sum(scores > true_val)
    n_lthan = np.sum(scores < true_val)
    if n_gthan + n_lthan == 0:
        return 0.0
    return n_gthan / (n_gthan + n_lthan)


print("=" * 80)
print("CALCULATING CONVERGENCE SCORES")
print("=" * 80)

conversation_convergence = []

for conv_id, group_data in df.groupby('conversation_id'):
    group_data = group_data.sort_values('rumble_number')

    if len(group_data) < 3:
        continue

    result = {
        'conversation_id': conv_id,
        'context_type': group_data['context_type'].iloc[0],
        'n_rumbles': len(group_data)
    }

    for feat_name in feature_names:
        result[f'convergence_{feat_name}'] = calculate_convergence(group_data, feat_name)

    result['convergence_overall'] = calculate_overall_convergence(group_data, feature_names)

    # Store raw Euclidean distances between adjacent rumbles for debugging
    feature_matrix = group_data[feature_names].values
    successive_distances = np.sqrt(np.sum(np.diff(feature_matrix, axis=0)**2, axis=1))
    result['adjacent_diffs'] = ';'.join([f'{d:.0f}' for d in successive_distances])

    conversation_convergence.append(result)

convergence_df = pd.DataFrame(conversation_convergence).dropna()

print(f"Successfully calculated convergence for {len(convergence_df)} conversations\n")

# Filter to single context if specified
if FILTER_CONTEXT is not None:
    convergence_df = convergence_df[convergence_df['context_type'] == FILTER_CONTEXT].copy()
    df = df[df['conversation_id'].isin(convergence_df['conversation_id'])].copy()
    print(f"Filtered to context '{FILTER_CONTEXT}': {len(convergence_df)} conversations\n")
    print("Per-conversation details:")
    for _, row in convergence_df.sort_values('conversation_id').iterrows():
        print(f"  {row['conversation_id']} (n={int(row['n_rumbles'])}): "
              f"convergence={row['convergence_overall']:.4f}, diffs=[{row['adjacent_diffs']}]")
    print()

# Get convergence columns
convergence_cols = [col for col in convergence_df.columns if col.startswith('convergence_')]

# Calculate mean convergence BY CONTEXT TYPE
print("=" * 80)
print("TRUE CONVERGENCE RATES BY CONTEXT TYPE")
print("=" * 80)

true_convergence_by_class = {}
for context_type in sorted(convergence_df['context_type'].unique()):
    class_data = convergence_df[convergence_df['context_type'] == context_type]
    print(f"\n{context_type.upper()} (n={len(class_data)} conversations):")
    true_convergence_by_class[context_type] = {}

    for col in convergence_cols:
        value = agg_func(class_data[col].values)
        true_convergence_by_class[context_type][col] = value
        if col == 'convergence_overall':
            print(f"  Overall convergence: {value:.4f}")


# ============================================================================
# PERMUTATION TESTS BY CONTEXT TYPE
# ============================================================================
print("\n" + "=" * 80)
print("PERMUTATION TESTS BY CONTEXT TYPE")
print("=" * 80)

n_permutations = 200
p_values_order_by_class = {}
p_values_order_std_by_class = {}
perm_results_order_by_class = {}

for context_type in sorted(convergence_df['context_type'].unique()):
    print(f"\n### {context_type.upper()} ###")

    class_conv_ids = convergence_df[convergence_df['context_type'] == context_type]['conversation_id'].values
    class_df = df[df['conversation_id'].isin(class_conv_ids)]

    if len(class_conv_ids) < 3:
        print(f"  Skipping - only {len(class_conv_ids)} conversations")
        continue

    print(f"  {len(class_conv_ids)} conversations")
    print(f"  Running order-shuffled permutation test...")

    perm_results_order = {col: [] for col in convergence_cols}

    for perm in range(n_permutations):
        if (perm + 1) % 50 == 0:
            print(f"    Permutation {perm + 1}/{n_permutations}")

        df_shuffled = class_df.copy()
        for conv_id, group_data in df_shuffled.groupby('conversation_id'):
            idx = group_data.index
            seed = stable_seed(str(conv_id) + str(perm) + str(PERM_SEED))
            df_shuffled.loc[idx, feature_names] = (
                df_shuffled.loc[idx, feature_names]
                .sample(frac=1, random_state=seed)
                .values
            )

        shuffled_convergence = []
        for conv_id, group_data in df_shuffled.groupby('conversation_id'):
            group_data = group_data.sort_values('rumble_number')
            if len(group_data) < 3:
                continue

            result = {}
            for feat_name in feature_names:
                result[f'convergence_{feat_name}'] = calculate_convergence(group_data, feat_name)
            result['convergence_overall'] = calculate_overall_convergence(group_data, feature_names)
            shuffled_convergence.append(result)

        shuffled_df = pd.DataFrame(shuffled_convergence).dropna()

        for col in convergence_cols:
            perm_results_order[col].append(shuffled_df[col].tolist())

    p_values_order = {}
    p_values_order_standard = {}
    for col in convergence_cols:
        true_val = true_convergence_by_class[context_type][col]
        p_values_order[col] = np.mean([gt_no_ties(scores, true_val) for scores in perm_results_order[col]])
        p_values_order_standard[col] = np.mean([agg_func(scores) >= true_val for scores in perm_results_order[col]])

    p_values_order_by_class[context_type] = p_values_order
    p_values_order_std_by_class[context_type] = p_values_order_standard
    perm_results_order_by_class[context_type] = perm_results_order

    print(f"    Overall p-value (gt_no_ties): {p_values_order['convergence_overall']:.4f}")
    print(f"    Overall p-value (standard):   {p_values_order_standard['convergence_overall']:.4f}")


# ============================================================================
# CONVERSATION-SHUFFLED PERMUTATION TEST (global - across all contexts)
# ============================================================================
print("\n" + "=" * 80)
print("CONVERSATION-SHUFFLED PERMUTATION TEST (global)")
print("=" * 80)

# Pre-build conv_id -> context_type lookup; restrict to conversations with valid convergence
conv_to_context = dict(zip(convergence_df['conversation_id'], convergence_df['context_type']))
valid_conv_ids = set(convergence_df['conversation_id'].values)
valid_df = df[df['conversation_id'].isin(valid_conv_ids)]

context_types = sorted(convergence_df['context_type'].unique())
perm_results_conv_by_class = {ct: [] for ct in context_types}

print(f"  Shuffling across all {len(valid_conv_ids)} conversations...")
print(f"  Running {n_permutations} permutations...")

for perm in range(n_permutations):
    if (perm + 1) % 50 == 0:
        print(f"    Permutation {perm + 1}/{n_permutations}")

    # Shuffle features at each position across ALL conversations
    df_conv_shuffled = valid_df.copy()
    for rumble_num in df_conv_shuffled['rumble_number'].unique():
        mask = df_conv_shuffled['rumble_number'] == rumble_num
        idx = df_conv_shuffled.loc[mask].index
        seed = stable_seed(str(rumble_num) + str(perm) + 'conv' + str(PERM_SEED))
        df_conv_shuffled.loc[idx, feature_names] = (
            df_conv_shuffled.loc[idx, feature_names]
            .sample(frac=1, random_state=seed)
            .values
        )

    # Calculate convergence per conversation, group by original context
    perm_scores_by_context = {ct: [] for ct in context_types}
    for conv_id, group_data in df_conv_shuffled.groupby('conversation_id'):
        if conv_id not in conv_to_context:
            continue
        ct = conv_to_context[conv_id]
        group_data = group_data.sort_values('rumble_number')
        if len(group_data) < 3:
            continue
        conv_rate = calculate_overall_convergence(group_data, feature_names)
        if not np.isnan(conv_rate):
            perm_scores_by_context[ct].append(conv_rate)

    for ct in context_types:
        if len(perm_scores_by_context[ct]) > 0:
            perm_results_conv_by_class[ct].append(perm_scores_by_context[ct])

# Compute p-values per class (both methods)
p_values_conv_by_class = {}
p_values_conv_std_by_class = {}

for context_type in context_types:
    true_val = true_convergence_by_class[context_type]['convergence_overall']
    if len(perm_results_conv_by_class[context_type]) > 0:
        p_values_conv_by_class[context_type] = np.mean(
            [gt_no_ties(scores, true_val) for scores in perm_results_conv_by_class[context_type]])
        p_values_conv_std_by_class[context_type] = np.mean(
            [agg_func(scores) >= true_val for scores in perm_results_conv_by_class[context_type]])
        print(f"  {context_type}: p (gt_no_ties)={p_values_conv_by_class[context_type]:.4f}, "
              f"p (standard)={p_values_conv_std_by_class[context_type]:.4f}")


print("\n" + "=" * 80)
print("SUMMARY: P-VALUES BY CONTEXT TYPE")
print("=" * 80)
for context_type in sorted(p_values_order_by_class.keys()):
    print(f"\n{context_type}:")
    print(f"  Order-shuffled:  gt_no_ties={p_values_order_by_class[context_type]['convergence_overall']:.4f}, "
          f"standard={p_values_order_std_by_class[context_type]['convergence_overall']:.4f}")
    if context_type in p_values_conv_by_class:
        print(f"  Conv-shuffled:   gt_no_ties={p_values_conv_by_class[context_type]:.4f}, "
              f"standard={p_values_conv_std_by_class[context_type]:.4f}")


# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

results_by_class = []
for context_type in sorted(true_convergence_by_class.keys()):
    if context_type not in p_values_order_by_class:
        continue

    n_conv = len(convergence_df[convergence_df['context_type'] == context_type])
    row = {
        'context_type': context_type,
        'n_conversations': n_conv,
        'overall_convergence': true_convergence_by_class[context_type]['convergence_overall'],
        'p_order_gt_no_ties': p_values_order_by_class[context_type]['convergence_overall'],
        'p_order_standard': p_values_order_std_by_class.get(context_type, {}).get('convergence_overall', np.nan),
        'p_conv_gt_no_ties': p_values_conv_by_class.get(context_type, np.nan),
        'p_conv_standard': p_values_conv_std_by_class.get(context_type, np.nan),
    }
    results_by_class.append(row)

results_by_class_df = pd.DataFrame(results_by_class)
results_file = output_dir / f'convergence_by_context_rumbles_{FEAT}_{overlap_tag}{truncate_tag}{median_tag}.csv'
results_by_class_df.to_csv(results_file, index=False, float_format='%.2f')
print(f"  Saved: {results_file}")

convergence_file = output_dir / f'convergence_by_conversation_rumbles_{FEAT}_{overlap_tag}{truncate_tag}{median_tag}.csv'
convergence_df.sort_values('context_type').to_csv(convergence_file, index=False, float_format='%.2f')
print(f"  Saved: {convergence_file}")


# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================


print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

# Save plots directly under the output directory (no separate plots/ subfolder)
figures_dir = output_dir
print(f"Figures directory: {figures_dir}\n")

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Calculate statistics for bar plot
from matplotlib.patches import Patch

context_stats = []

for context_type in sorted(convergence_df['context_type'].unique()):
    context_data = convergence_df[convergence_df['context_type'] == context_type]
    n = len(context_data)

    mean_overall = agg_func(context_data['convergence_overall'].values)
    se_overall = context_data['convergence_overall'].sem()

    p_order = p_values_order_std_by_class.get(context_type, {}).get('convergence_overall', np.nan)
    p_conv = p_values_conv_std_by_class.get(context_type, np.nan)
    
    context_stats.append({
        'context_type': context_type,
        'mean': mean_overall,
        'se': se_overall,
        'n': n,
        'p_value': p_order,
        'p_order': p_order,
        'p_conv': p_conv,
    })

context_stats_df = pd.DataFrame(context_stats)

# Custom sort order: Cadenced, Contact, Greeting, then alphabetical
priority_order = ['Cadenced', 'Contact', 'Greeting']
def sort_key(ct):
    if ct in priority_order:
        return (0, priority_order.index(ct))
    else:
        return (1, ct)
context_stats_df['_sort_key'] = context_stats_df['context_type'].apply(sort_key)
context_stats_df = context_stats_df.sort_values('_sort_key').drop(columns=['_sort_key']).reset_index(drop=True)

# ============================================================================
# Bar plot: combined, grouped bars with error bars
# ============================================================================
print("Creating combined bar plot...")

SIG_COLOR = '#129c7b'
NONSIG_COLOR = 'gray'

def bar_color(row):
    p_o = row['p_order']
    p_c = row['p_conv']
    sig = (not pd.isna(p_o) and p_o < 0.05) or (not pd.isna(p_c) and p_c < 0.05)
    return SIG_COLOR if sig else NONSIG_COLOR

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_facecolor('white')
ax.grid(False)
ax.tick_params(which='both', direction='out', length=4, color='black', 
               labelcolor='black', bottom=True, left=True)
x_centers = np.arange(len(context_stats_df))
bar_width = 0.6

# Single bar per context type, green if any test significant, gray otherwise
bar_colors = [bar_color(row) for _, row in context_stats_df.iterrows()]
ax.bar(x_centers,
       context_stats_df['mean'],
       width=bar_width,
       yerr=context_stats_df['se'],
       color=bar_colors,
       alpha=0.7,
       edgecolor='black',
       linewidth=1.2,
       capsize=5,
       error_kw={'linewidth': 1.5, 'ecolor': 'black'})

# Add significance stars above bars
# Vermillion stars = order-shuffled, sky blue stars = conv-shuffled (Okabe-Ito colorblind-safe)
color_order_star = '#D55E00'   # Vermillion
color_conv_star  = '#0072B2'   # Deep blue

for i, (idx, row) in enumerate(context_stats_df.iterrows()):
    y_base = row['mean'] + row['se'] + 0.01 if row['mean'] >= 0 else row['mean'] - row['se'] - 0.01
    va = 'bottom' if row['mean'] >= 0 else 'top'

    # Order-shuffled stars (top row)
    p_o = row['p_order']
    if not pd.isna(p_o) and p_o < 0.05:
        star_o = '***' if p_o < 0.001 else ('**' if p_o < 0.01 else '*')
        ax.text(i, y_base, star_o, ha='center', va=va,
                fontsize=18, fontweight='bold', color=color_order_star)

    # Conv-shuffled stars (second row, offset down/up)
    p_c = row['p_conv']
    offset = 0.03 if row['mean'] >= 0 else -0.03
    if not pd.isna(p_c) and p_c < 0.05:
        star_c = '***' if p_c < 0.001 else ('**' if p_c < 0.01 else '*')
        ax.text(i, y_base + offset, star_c, ha='center', va=va,
                fontsize=18, fontweight='bold', color=color_conv_star)

# Print sample sizes to terminal
print("\nBar plot sample sizes:")
for _, row in context_stats_df.iterrows():
    print(f"  {row['context_type']:<20} n={int(row['n'])}")

ax.set_xlabel('Context Type', fontsize=12, fontweight='bold')
ax.set_ylabel('Convergence Rate', fontsize=12, fontweight='bold')
ax.set_xticks(x_centers)
ax.set_xticklabels(context_stats_df['context_type'], rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_ylim(top=.85)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=SIG_COLOR,
           markersize=10, markeredgecolor='black', label='Significant (p < 0.05)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=NONSIG_COLOR,
           markersize=10, markeredgecolor='black', label='n.s.'),
    Line2D([0], [0], color='none', label=''),  # Spacer
    Line2D([0], [0], marker='*', color=color_order_star, linestyle='None',
           markersize=12, label='Order-shuffled test'),
    Line2D([0], [0], marker='*', color=color_conv_star, linestyle='None',
           markersize=12, label='Exchange-shuffled test'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=1)

plt.tight_layout()
bar_plot_file = figures_dir / f'convergence_barplot_rumbles_{FEAT}_{overlap_tag}{truncate_tag}{median_tag}.png'
plt.savefig(bar_plot_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {bar_plot_file}")
plt.close()



# ============================================================================
# Permutation mean distribution: histogram of perm means vs true mean
# ============================================================================
print("Creating permutation mean distribution plot...")

# Two-column layout: left = Contact, Greeting, Coo; right = Cadenced, Lets go, Little greeting
all_context_types = convergence_df['context_type'].unique()
left_col = [ct for ct in ['Contact', 'Greeting', 'Coo'] if ct in all_context_types]
right_col = [ct for ct in ['Cadenced', 'Lets go', 'Little greeting'] if ct in all_context_types]
n_rows_perm = max(len(left_col), len(right_col))
context_grid = []
for row_idx in range(n_rows_perm):
    context_grid.append(left_col[row_idx] if row_idx < len(left_col) else None)
    context_grid.append(right_col[row_idx] if row_idx < len(right_col) else None)

fig, axes = plt.subplots(n_rows_perm, 2, figsize=(14, 3.5 * n_rows_perm), squeeze=False)

for grid_idx, ct in enumerate(context_grid):
    row_idx = grid_idx // 2
    col_idx = grid_idx % 2
    ax = axes[row_idx, col_idx]

    if ct is None:
        ax.set_visible(False)
        continue

    true_scores = convergence_df[convergence_df['context_type'] == ct]['convergence_overall'].values
    true_mean = agg_func(true_scores)

    # Permutation means: one mean per permutation
    order_perm_lists = perm_results_order_by_class.get(ct, {}).get('convergence_overall', [])
    conv_perm_lists = perm_results_conv_by_class.get(ct, [])

    order_means = [agg_func(scores) for scores in order_perm_lists if len(scores) > 0]
    conv_means  = [agg_func(scores) for scores in conv_perm_lists  if len(scores) > 0]

    # Shared bin edges across both distributions
    all_vals = order_means + conv_means
    if len(all_vals) > 0:
        bin_min = min(all_vals + [true_mean])
        bin_max = max(all_vals + [true_mean])
        bin_range = bin_max - bin_min if bin_max > bin_min else 1.0
        bins = np.linspace(bin_min - 0.05 * bin_range, bin_max + 0.05 * bin_range, 30)
    else:
        bins = 30

    # Color-blind friendly palette (Okabe-Ito)
    color_order_perm = '#E69F00'  # Orange
    color_true = '#0072B2'        # Blue
    color_conv_perm = '#009E73'   # Green

    if len(order_means) > 0:
        ax.hist(order_means, bins=bins, color=color_order_perm, alpha=0.6, label='Order-shuffled', edgecolor='white', linewidth=0.5)
    if len(conv_means) > 0:
        ax.hist(conv_means,  bins=bins, color=color_conv_perm, alpha=0.6, label='Conv-shuffled',  edgecolor='white', linewidth=0.5)

    # Vertical line at true mean
    ax.axvline(x=true_mean, color=color_true, linewidth=2.2, linestyle='-', label=f'True {agg_name} ({true_mean:.3f})')

    ax.set_title(f'{ct} (n={len(true_scores)})', fontsize=11, fontweight='bold')
    ax.set_xlabel(f'{agg_name.capitalize()} convergence rate', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.legend(fontsize=7.5, loc='best')
    ax.grid(axis='y', alpha=0.3)

#fig.suptitle('Permutation Mean Distributions vs True Mean', fontsize=14, fontweight='bold')
plt.tight_layout()
permhist_plot_file = figures_dir / f'convergence_permhist_rumbles_{FEAT}_{overlap_tag}{truncate_tag}{median_tag}.png'
plt.savefig(permhist_plot_file, dpi=300, bbox_inches='tight')
print(f"  Saved: {permhist_plot_file}")
plt.close()

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"All figures saved to: {figures_dir}\n")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Dataset totals
total_convos = df['conversation_id'].nunique()
total_rumbles = len(df)
rpc = df.groupby('conversation_id').size()
print(f"\nDataset:")
print(f"  Conversations : {total_convos}")
print(f"  Rumbles       : {total_rumbles}")
print(f"  Rumbles/conv  : {rpc.mean():.1f} ± {rpc.sem():.2f} (range {rpc.min()}–{rpc.max()})")

# Per-context breakdown
print(f"\n{'Context':<20} {'Convos':>6} {'Rumbles':>8} {'Mean R/conv':>12} {'Convergence':>12} {'SE':>8} {'p_order':>8} {'p_conv':>8} {'sig_order':>10} {'sig_conv':>9}")
print("-" * 105)
for _, row in context_stats_df.iterrows():
    ct = row['context_type']
    ct_df = df[df['context_type'] == ct]
    n_convos = ct_df['conversation_id'].nunique()
    n_rumbles = len(ct_df)
    mean_rpc = ct_df.groupby('conversation_id').size().mean()
    p_o = row['p_order']
    p_c = row['p_conv']
    sig_o = '***' if p_o < 0.001 else ('**' if p_o < 0.01 else ('*' if p_o < 0.05 else 'n.s.'))
    sig_c = '***' if p_c < 0.001 else ('**' if p_c < 0.01 else ('*' if p_c < 0.05 else 'n.s.')) if pd.notna(p_c) else 'n/a'
    print(f"  {ct:<18} {n_convos:>6} {n_rumbles:>8} {mean_rpc:>12.1f} {row['mean']:>12.4f} {row['se']:>8.4f} {p_o:>8.3f} {p_c:>8.3f} {sig_o:>10} {sig_c:>9}")

print("-" * 105)
print(f"  {'TOTAL':<18} {total_convos:>6} {total_rumbles:>8}")

print("\nConversations included per context type:")
for ct in sorted(df['context_type'].unique()):
    ids = sorted(df[df['context_type'] == ct]['conversation_id'].unique())
    print(f"\n  {ct} (n={len(ids)}):")
    print(f"    {'ID':<30} {'N rumbles':>10} {'Convergence':>12} {'Pairwise distances'}")
    print(f"    {'-'*80}")
    for cid in ids:
        n = len(df[df['conversation_id'] == cid])
        conv_row = convergence_df[convergence_df['conversation_id'] == cid]
        if len(conv_row) > 0:
            conv_val = conv_row['convergence_overall'].iloc[0]
            diffs = conv_row['adjacent_diffs'].iloc[0]
        else:
            conv_val = float('nan')
            diffs = 'n/a'
        print(f"    {cid:<30} {n:>10} {conv_val:>12.4f}  {diffs}")

print(f"\nAll results saved to: {output_dir}")
