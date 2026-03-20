import os
RUMBLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rumbles')

FEATS = ['loglinpow-30-80', 'acoustic-contour-feats']
AGS = ['single']
BALANCE_CLASSES = True
N_FOLDS = 20
TARGET = 'context'
SHOW_PLOTS = False
OTHERS_FAMS = 'all'
SMALL_CLASS_THRESH = 5
RECOMPUTE = False
EXCLUDE_CALVES = True
N_ITER = 10000
N_FEAT_IMPORTANCES = 3
ACOUSTIC_N_PCS = 10
ACOUSTIC_N_FFT_PEAKS = 4
from sklearn.model_selection import LeaveOneGroupOut
import json
from scipy.signal import resample
from collections import Counter
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from utils import filter_metadata_csv, save_and_plot_results
from scipy import stats
from scipy.stats import skew
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import random

'''
# Example usage:

# EB single -> EB single
python3.11 predict_context.py \
    --dataset EB --subset single \
    --exp-name EB_single

# EB single -> EB single with held-out callers
python3.11 predict_context.py \
    --dataset EB --subset single \
    --test-on-held-out-callers \
    --exp-name EB_single-loo

# nonEB single -> nonEB single
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --exp-name nonEB_single

# nonEB single -> nonEB single with held-out callers
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --test-on-held-out-callers \
    --exp-name nonEB_single-loo

# EB chorus -> EB chorus    
python3.11 predict_context.py \
    --dataset EB --subset chorus \
    --exp-name EB_chorus
    
# nonEB chorus -> nonEB chorus    
python3.11 predict_context.py \
    --dataset nonEB --subset chorus \
    --exp-name nonEB_chorus

# nonEB single -> EB single
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --dataset-test EB --subset-test single \
    --exp-name EB_single_train_nonEB_single

# EB single -> nonEB single
python3.11 predict_context.py \
    --dataset EB --subset single \
    --dataset-test nonEB --subset-test single \
    --exp-name nonEB_single_train_EB_single

# nonEB chorus -> EB chorus
python3.11 predict_context.py \
    --dataset nonEB --subset chorus \
    --dataset-test EB --subset-test chorus \
    --exp-name EB_chorus_train_nonEB_chorus 

# EB chorus -> nonEB chorus
python3.11 predict_context.py \
    --dataset EB --subset chorus \
    --dataset-test nonEB --subset-test chorus \
    --exp-name nonEB_chorus_train_EB_chorus 

# nonEB single -> EB chorus
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --dataset-test EB --subset-test chorus \
    --exp-name EB_chorus_train_nonEB_single 

# EB single -> nonEB chorus
python3.11 predict_context.py \
    --dataset EB --subset single \
    --dataset-test nonEB --subset-test chorus \
    --exp-name nonEB_chorus_train_EB_single 

# nonEB single -> nonEB chorus
python3.11 predict_context.py \
    --dataset nonEB --subset single \
    --dataset-test nonEB --subset-test chorus \
    --exp-name nonEB_chorus_train_nonEB_single 

# EB single -> EB chorus
python3.11 predict_context.py \
    --dataset EB --subset single \
    --dataset-test EB --subset-test chorus \
    --exp-name EB_chorus_train_EB_single 


    



'''
random.seed(1)
np.random.seed(1)

near_emptys = []
ooncs = []
aetcs = []

conv_level_feat_names = ['n-calls', 'overlap-ratio', 'latency']

def compare_acc_to_chance(preds, gts_test, gts_train):
    accuracy = (preds == gts_test).mean()
    majority_class_accuracy = np.max(np.bincount(gts_train)) / len(gts_train)
    class_proportions = np.bincount(gts_train) / len(gts_train)
    prop_accuracy = np.sum(class_proportions ** 2)

    print(f'Accuracy: {accuracy:.4f}')
    print('\nvs Bernoulli:')

    p_value_prop = stats.binomtest(int(accuracy * len(gts_test)), n=len(gts_test), p=prop_accuracy).pvalue
    p_value_majority = stats.binomtest(int(accuracy * len(gts_test)), n=len(gts_test), p=majority_class_accuracy).pvalue

    for bname, bacc, pval in zip(['Prop-rand', 'Always-majority'],
                                  [prop_accuracy, majority_class_accuracy],
                                  [p_value_prop, p_value_majority]):
        print(f"\t{bname}: {bacc:.4f}, p-value: {pval:.6f}")

    print('\nvs MC:')

    rng = np.random.default_rng(222)
    fake_preds_gt = np.stack([rng.permutation(gts_test) for _ in range(10000)])
    fake_accs_gt = (fake_preds_gt == gts_test).mean(axis=1)
    p_value_gt_dist = (fake_accs_gt >= accuracy).mean()

    fake_preds_pred = np.stack([rng.permutation(preds) for _ in range(10000)])
    fake_accs_pred = (fake_preds_pred == gts_test).mean(axis=1)
    p_value_pred_dist = (fake_accs_pred >= accuracy).mean()

    for bname, fake_accs, pval in zip(['GT-distribution', 'pred-distribution'],
                                       [fake_accs_gt, fake_accs_pred],
                                       [p_value_gt_dist, p_value_pred_dist]):
        print(f"\t{bname}: {fake_accs.mean():.4f}, p-value: {pval:.5f}")

    f1 = f1_score(gts_test, preds, average='macro')

    return accuracy, f1, p_value_prop, p_value_majority, p_value_gt_dist, fake_accs_pred, p_value_pred_dist, prop_accuracy, majority_class_accuracy


def save_results_to_table(experiment_name, train_data, test_data, features, aggregation,
                         accuracy, f1_macro, fake_accs_pred, p_value_pred_dist, prop_accuracy, majority_accuracy, n_samples, n_classes, results_file):
    from datetime import datetime

    train_data_clean = train_data.replace(RUMBLES_DIR + '/', '')
    test_data_clean = test_data.replace(RUMBLES_DIR + '/', '')

    result_row = {
        'experiment_name': experiment_name,
        'train_data': train_data_clean,
        'test_data': test_data_clean,
        'n_samples': n_samples,
        'n_classes': n_classes,
        'baseline_prop': round(prop_accuracy, 4),
        'baseline_majority': round(majority_accuracy, 4),
        'pred_distribution_mean': round(np.mean(fake_accs_pred), 6),
        'f1_macro': round(f1_macro, 4),
        'accuracy': round(accuracy, 4),
        'p_value_pred_distribution': round(p_value_pred_dist, 6),
    }

    results_df = pd.DataFrame([result_row])
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Results saved to {results_file}")
    return results_df


def resample_contour(contour, target_len):
    x_old = np.linspace(0, 1, len(contour))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, contour)

def compute_log_power_in_bands(x, n_freq, bandwidth, sr=None):
      
    if isinstance(x, str):
        assert sr is None
        y_sp, sr = librosa.load(x, sr=None)
    else:
        assert isinstance(x, np.ndarray)
        assert sr is not None
        y_sp = x
    
    assert sr == 16000, f"Expected sample rate 16000, got {sr}"
    
    n_fft = 800
    if len(y_sp) < n_fft:
        print(f"ERROR: Audio too short for n_fft={n_fft}: {len(y_sp)} samples")
        return None
    
    win_length = 512
    hop = 128
    hz_per_row = sr / n_fft
    rows_per_band = int(bandwidth / hz_per_row)
    
    try:
        S = np.abs(librosa.stft(y_sp, n_fft=n_fft, hop_length=hop, win_length=win_length))**2      
        S_db = librosa.power_to_db(S, ref=np.max)
        max_freq_bins = n_freq * rows_per_band
        
        if max_freq_bins > S_db.shape[0]:
            print(f"ERROR: Not enough frequency bins: need {max_freq_bins}, have {S_db.shape[0]}")
            return None
        
        grouped_db = S_db[:max_freq_bins].reshape(n_freq, rows_per_band, -1)
        spf = grouped_db.mean(axis=(1, 2))
        return spf
        
    except Exception as e:
        print(f"ERROR: Exception in compute_log_power_in_bands: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_predictions_to_file(origX, y_true, y_pred, le, expname, metadata=None):
    """
    Save predictions to a text file with filename, caller name, true label, and predicted label.
    
    Parameters:
    -----------
    origX : DataFrame
        Original feature DataFrame with file paths as index
    y_true : array
        True encoded labels
    y_pred : array
        Predicted encoded labels
    le : LabelEncoder
        Label encoder to convert back to class names
    expname : str
        Experiment name for output file
    metadata : DataFrame, optional
        Metadata containing CallerName information
    """
    # Decode labels back to original class names
    y_true_names = le.inverse_transform(y_true)
    y_pred_names = le.inverse_transform(y_pred)
    
    # Get filenames from index
    filenames = [os.path.basename(fp) for fp in origX.index]
    
    # Get caller names if metadata provided
    caller_names = []
    if metadata is not None:
        # Set index to SndFile for easy lookup
        if 'SndFile' in metadata.columns:
            metadata_indexed = metadata.set_index('SndFile')
        else:
            metadata_indexed = metadata
        
        for fp in origX.index:
            file_id = os.path.basename(fp)
            if file_id in metadata_indexed.index:
                if 'CallerName' in metadata_indexed.columns:
                    caller = metadata_indexed.loc[file_id, 'CallerName']
                elif 'Caller' in metadata_indexed.columns:
                    caller = metadata_indexed.loc[file_id, 'Caller']
                else:
                    caller = 'Unknown'
                
                # Handle cases where caller might be a list/array
                if isinstance(caller, (list, np.ndarray, pd.Series)):
                    caller = caller[0] if len(caller) > 0 else 'Unknown'
                elif pd.isna(caller):
                    caller = 'Unknown'
                caller_names.append(str(caller))
            else:
                caller_names.append('Unknown')
    else:
        caller_names = ['N/A'] * len(filenames)
    
    # Create output file
    output_file = f'context_prediction_output/{expname}/{expname}_predictions.txt'
    
    with open(output_file, 'w') as f:
        f.write(f"{'Filename':<60} {'CallerName':<20} {'True_Label':<30} {'Predicted_Label':<30} {'Correct'}\n")
        f.write("=" * 145 + "\n")
        
        for filename, caller, true_label, pred_label in zip(filenames, caller_names, y_true_names, y_pred_names):
            correct = "✓" if true_label == pred_label else "✗"
            f.write(f"{filename:<60} {caller:<20} {true_label:<30} {pred_label:<30} {correct}\n")
    
    
    # Also save as CSV for easier analysis
    csv_file = f'context_prediction_output/{expname}/{expname}_predictions.csv'
    predictions_df = pd.DataFrame({
        'Filename': filenames,
        'CallerName': caller_names,
        'True_ContextTypeNew': y_true_names,
        'Predicted_ContextTypeNew': y_pred_names,
        'Correct': y_true_names == y_pred_names
    })
    predictions_df.to_csv(csv_file, index=False)
    
    return predictions_df


def db_mel_spec_from_fp(fp, recompute=False, ARGS = 'none'):
    cache_fp = fp.replace('.wav', '_powerbands.npy')
    recompute = RECOMPUTE

    if os.path.exists(cache_fp) and not recompute:
        return np.load(cache_fp)

    if not os.path.exists(fp):
        if os.path.exists(cache_fp):
            return np.load(cache_fp)
        return None

    y, sr = librosa.load(fp, sr=None)
    
    assert sr == 16000
    n_fft = int(sr * 0.350/8); hop_length = int(n_fft * 0.1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hamming', n_mels=13, fmin=80, fmax=2000, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    
    np.save(cache_fp, S_db)
    return S_db

def aggregate_call_feats_across_conv(features_by_call, feat_names, ag_methods):
    agged_features_by_name = {}
    if 'mean' in ag_methods:
        mean_feats = features_by_call.mean(axis=0)
        for fn, fv in zip(feat_names, mean_feats):
            agged_features_by_name[f'mean-{fn}'] = fv
    if 'std' in ag_methods:
        std_feats = features_by_call.std(axis=0)
        for fn, fv in zip(feat_names, std_feats):
            agged_features_by_name[f'std-{fn}'] = fv
    if 'first' in ag_methods:
        for fn, fv in zip(feat_names, features_by_call[0]):
            agged_features_by_name[f'first-{fn}'] = fv
    if 'last' in ag_methods:
        for fn, fv in zip(feat_names, features_by_call[-1]):
            agged_features_by_name[f'last-{fn}'] = fv
    return agged_features_by_name

def acoustic_features_from_contours(X, n_pcs=10, n_fft_peaks=4,
                                   include_mean=True, include_median=True, include_skewness=True,
                                   include_min=True, include_iqr=True, include_fft=True):
    N, nf, nt = X.shape

    pca_input = X.transpose(0, 2, 1).reshape(-1, nf)  # (N*nt, nf)
    scaler = RobustScaler()
    pca_input = scaler.fit_transform(pca_input)
    pca = PCA(n_components=n_pcs, random_state=999)
    pca_output = pca.fit_transform(pca_input)  # (N*nt, n_pcs)
    X_work = pca_output.reshape(N, nt, n_pcs).transpose(0, 2, 1)  # (N, n_pcs, nt)
    n_components = n_pcs
    component_names = [f'pca_{i}' for i in range(n_pcs)]


    
    # Collect enabled statistical features
    stat_features_list = []
    stat_feat_names = []
    
    if include_mean:
        mean_vals = np.mean(X_work, axis=2)
        stat_features_list.append(mean_vals)
        stat_feat_names.append('mean')
    
    if include_median:
        median_vals = np.median(X_work, axis=2)
        stat_features_list.append(median_vals)
        stat_feat_names.append('median')
    
    if include_skewness:
        skewness_vals = skew(X_work, axis=2)
        stat_features_list.append(skewness_vals)
        stat_feat_names.append('skewness')
    
    if include_min:
        min_vals = np.min(X_work, axis=2)
        stat_features_list.append(min_vals)
        stat_feat_names.append('min')
    
    if include_iqr:
        stat_extent = np.percentile(X_work, 75, axis=2) - np.percentile(X_work, 25, axis=2)
        stat_features_list.append(stat_extent)
        stat_feat_names.append('iqr')
    
    # Combine statistical features
    feature_arrays = []
    base_feat_names = []
    
    if stat_features_list:
        stat_feats = np.stack(stat_features_list, axis=2)  # (N, n_components, n_stat_feats)
        feature_arrays.append(stat_feats)
        base_feat_names.extend(stat_feat_names)

    # Calculate FFT features if enabled
    if include_fft and n_fft_peaks > 0:
        freqs = np.fft.fftfreq(nt)
        freq_domain_components = np.abs(fft(X_work, axis=2))
        freq_domain_peaks = np.argpartition(-freq_domain_components, n_fft_peaks, axis=2)[:, :, :n_fft_peaks]
        freq_domain_peak_vals = np.take_along_axis(freq_domain_components, freq_domain_peaks, axis=2)
        freq_domain_peak_freqs = np.array([[freqs[row] for row in mat] for mat in freq_domain_peaks])
        fft_feats = np.concatenate([freq_domain_peak_freqs, freq_domain_peak_vals], axis=2)  # (N, n_components, 2*n_fft_peaks)
        
        feature_arrays.append(fft_feats)
        fft_feat_names = [f'fft_freq_{i+1}' for i in range(n_fft_peaks)] + [f'fft_mag_{i+1}' for i in range(n_fft_peaks)]
        base_feat_names.extend(fft_feat_names)
    
    # Check if any features are enabled
    if not feature_arrays:
        raise ValueError("At least one feature type must be enabled for acoustic contour features")
    
    
    # Combine all enabled features
    if len(feature_arrays) == 1:
        all_feats = feature_arrays[0]
    else:
        all_feats = np.concatenate(feature_arrays, axis=2)  # (N, n_components, total_features)
    
    # Flatten features and create column names
    n_feats_per_component = len(base_feat_names)
    dpoint_feats = all_feats.reshape(N, n_components * n_feats_per_component)
    dpoint_feat_names = [f'{comp_name}_{feat_name}' for comp_name in component_names for feat_name in base_feat_names]
    
    dpoint_feats_df = pd.DataFrame(dpoint_feats, columns=dpoint_feat_names)
    return dpoint_feats_df

def extract_conv_features(audio_fp, feats, recompute=False):
    fn_base = os.path.basename(audio_fp).removesuffix('.wav')
    call_feats = [x for x in feats if x not in conv_level_feat_names]
    single_split_feats, call_feat_names = extract_call_features(audio_fp, call_feats, recompute=recompute)
    if single_split_feats is None or call_feat_names is None:
        print(f"WARNING: Failed to extract features from {audio_fp}")
        return None
    if isinstance(single_split_feats, (list, np.ndarray)) and len(single_split_feats) == len(call_feat_names):
        features_by_name = {f'single-{k}': v for k, v in zip(call_feat_names, single_split_feats)}
    else:
        print(f"ERROR: Mismatch between features ({len(single_split_feats) if hasattr(single_split_feats, '__len__') else 'scalar'}) and names ({len(call_feat_names)})")
        return None
    return features_by_name

def extract_conv_dset_features(audio_fp_list, feats, ag_methods, recompute=False, ARGS='none'):
    if not recompute:
        print("Loading features from cache...")

    existing_files = [fp for fp in audio_fp_list if os.path.exists(fp)]
    if len(existing_files) == 0:
        print("No wav files found — attempting to load from cache...")
   
    indiv_convs_feats = [x for x in feats if x not in ['acoustic-contour-feats']]
    if len(indiv_convs_feats) == 0:
        features_by_fp = pd.DataFrame(index=audio_fp_list)
    else:
        features_by_fp = {}
        valid_features = {}

        for afp in audio_fp_list:
            try:
                features = extract_conv_features(afp, indiv_convs_feats, recompute=recompute)
                if features is not None and isinstance(features, dict) and len(features) > 0:
                    valid_features[afp] = features
                else:
                    print(f"Warning: No features extracted for {afp}")
            except Exception as e:
                print(f"Error extracting features from {afp}: {e}")
                continue

        if len(valid_features) == 0:
            print("ERROR: No valid features extracted from any files!")
            return pd.DataFrame(index=audio_fp_list)
        
        # Create DataFrame only from valid features
        features_by_fp = pd.DataFrame(valid_features).T
        
        # Reindex to include all original files (missing ones will have NaN)
        features_by_fp = features_by_fp.reindex(audio_fp_list)
    
    if 'acoustic-contour-feats' in feats:
        try:
            ac_feats = extract_acoustic_contour_conv_features(audio_fp_list, ag_methods, ARGS=ARGS)
           
            # Check if acoustic features are empty
            if ac_feats.empty or ac_feats.shape[1] == 0:
                print("WARNING: Acoustic contour features returned empty DataFrame")
                print("This usually means no valid conversation splits were found")
                # If we have other features, continue without acoustic features
                if not features_by_fp.empty and features_by_fp.shape[1] > 0:
                    print("Continuing with other features only")
                    return features_by_fp
                else:
                    print("No features extracted at all - returning empty DataFrame")
                    return pd.DataFrame(index=audio_fp_list)
            else:
                features_by_fp = pd.concat([features_by_fp, ac_feats], axis=1)
        except Exception as e:
            print(f"Error extracting acoustic contour features: {e}")
            if features_by_fp.empty or features_by_fp.shape[1] == 0:
                return pd.DataFrame(index=audio_fp_list)
    # Final check for empty features
    if features_by_fp.empty or features_by_fp.shape[1] == 0:
        print(f"ERROR: No features extracted for any files!")
        print(f"Feature types requested: {feats}")
        print(f"Files to process: {len(audio_fp_list)}")
        return pd.DataFrame(index=audio_fp_list)
    
    return features_by_fp

def extract_acoustic_contour_conv_features(audio_fp_list, ag_methods, level='conv', ARGS = 'none'):
    n_calls_per_conv = {}
    all_db_mel_specs_dict = {}
    
    for afp in audio_fp_list:
        data_dir = os.path.dirname(afp)
        fn_base = os.path.basename(afp).removesuffix('.wav')
        
        fn = f'{fn_base}.wav'
        if fn not in near_emptys:
            call_fp = os.path.join(data_dir, fn)
            try:
                S_db = db_mel_spec_from_fp(call_fp, ARGS=ARGS)
                if S_db is not None and len(S_db) > 0:
                    all_db_mel_specs_dict[fn] = S_db
                    n_calls_per_conv[afp] = 1
                else:
                    n_calls_per_conv[afp] = 0
            except Exception as e:
                print(f"Warning: Failed to extract mel spectrogram from {call_fp}: {e}")
                n_calls_per_conv[afp] = 0
        else:
            n_calls_per_conv[afp] = 0
    
    # Check if we have any valid spectrograms
    if len(all_db_mel_specs_dict) == 0:
        print("WARNING: No valid mel spectrograms found for acoustic contour features.")
        print("This usually means no valid conversation splits or single files were found.")
        return pd.DataFrame(index=audio_fp_list)
    
    # Get valid lengths and check for NaN
    valid_lengths = []
    for spec in all_db_mel_specs_dict.values():
        if hasattr(spec, 'shape') and len(spec.shape) > 1:
            length = spec.shape[1]
            if not np.isnan(length) and length > 0:
                valid_lengths.append(length)
    
    if len(valid_lengths) == 0:
        print("WARNING: No valid mel spectrograms with proper dimensions found.")
        return pd.DataFrame(index=audio_fp_list)
    
    target_len = int(np.median(valid_lengths))
    
    # Resample all spectrograms to target length
    resampled_specs = []
    for spec in all_db_mel_specs_dict.values():
        try:
            if hasattr(spec, 'shape') and len(spec.shape) > 1 and spec.shape[1] > 0:
                resampled = resample(spec, target_len, axis=1)
                if not np.isnan(resampled).any():
                    resampled_specs.append(resampled)
        except Exception as e:
            print(f"Warning: Error resampling spectrogram: {e}")
            continue
    
    if len(resampled_specs) == 0:
        print("WARNING: No spectrograms could be resampled successfully.")
        return pd.DataFrame(index=audio_fp_list)
    
    all_contours = np.stack(resampled_specs)
    
    # Extract acoustic features
    all_pardo_feats = acoustic_features_from_contours(
        all_contours,
        n_pcs=ACOUSTIC_N_PCS,
        n_fft_peaks=ACOUSTIC_N_FFT_PEAKS,
        include_mean=True,
        include_median=True,
        include_skewness=True,
        include_min=True,
        include_iqr=True,
        include_fft=True
    )
    
    
    # Map features back to conversations
    conv_split_points = np.cumsum(list(n_calls_per_conv.values()))
    starts = [0] + list(conv_split_points)[:-1]
    
    dset_level_feats = []
    conv_names = list(n_calls_per_conv.keys())
    
    for (start, end) in zip(starts, conv_split_points):
        if start < end and start < len(all_pardo_feats):
            # We have valid features for this conversation
            conv_features = all_pardo_feats.iloc[start:end]
            if len(conv_features) > 0:
                if ag_methods == ['single']:
                    # For single files, just use the features as-is
                    feature_dict = conv_features.iloc[0].to_dict()
                    dset_level_feats.append(feature_dict)
                else:
                    # For conversation splits, aggregate across calls
                    agg_features = aggregate_call_feats_across_conv(
                        conv_features.values, 
                        feat_names=conv_features.columns, 
                        ag_methods=ag_methods
                    )
                    dset_level_feats.append(agg_features)
            else:
                dset_level_feats.append({})
        else:
            dset_level_feats.append({})
    
    dset_level_feats_by_conv = pd.DataFrame(dset_level_feats, index=conv_names)
    return dset_level_feats_by_conv

def extract_call_features(audio_fp, feats, frame_mfcc_std=False, recompute=False):
    fn_base = os.path.basename(audio_fp).removesuffix('.wav')
    cache_dir = os.path.join(os.path.dirname(audio_fp), fn_base)
    call_feats = [ft for ft in feats if ft != 'acoustic-contour-feats']

    # If wav is missing, try to load all features from cache
    if not os.path.exists(audio_fp):
        all_cache_fps = [os.path.join(cache_dir, f'{ft}.npy') for ft in call_feats]
        if all(os.path.exists(cfp) for cfp in all_cache_fps):
            features, feat_names = [], []
            for ft, cfp in zip(call_feats, all_cache_fps):
                arr = np.load(cfp)
                features.append(arr)
                if ft.startswith('loglinpow'):
                    n_bands = int(ft.split('-')[1])
                    feat_names += [f'logpow_band{i}' for i in range(n_bands)]
            return np.concatenate(features), feat_names
        print(f"WARNING: Audio file does not exist and no cache found: {audio_fp}")
        return None, None

    # Check file size
    file_size = os.path.getsize(audio_fp)
    if file_size < 1000:
        print(f"  ERROR: File too small ({file_size} bytes)")
        return None, None

    try:
        y, sr = librosa.load(audio_fp, sr=None)
    except Exception as e:
        print(f"ERROR: Failed to load audio file {audio_fp}: {e}")
        return None, None
    
    # Check if audio has sufficient samples
    if len(y) == 0:
        print(f"  ERROR: Audio file is empty")
        return None, None
    
    if len(y) < 800:  # n_fft requirement
        print(f"  ERROR: Audio too short ({len(y)} samples < 800 required)")
        return None, None
    
    scale_by = 2**(16-1) * 0.7 / y.max()
    y *= scale_by
    y -= y.mean()
    
    if len(y)==0:
        return None, None
    os.makedirs(cache_dir, exist_ok=True)

    def load_or_save(cache_fp, compute_fn):
        if os.path.exists(cache_fp) and not recompute:
            return np.load(cache_fp)
        result = compute_fn()
        np.save(cache_fp, result)
        return result

    features = []
    feat_names = []
    for ft in feats:
        cache_fp = os.path.join(cache_dir, f'{ft}.npy')
        
        try:
            if ft.startswith('loglinpow'):
                parts = ft.split('-')
                n_bands = int(parts[1])
                bandwidth = int(parts[2])
                assert n_bands <= 50, f"Too many bands: {n_bands} > 50"
                
                loglin_feats = load_or_save(cache_fp, lambda: compute_log_power_in_bands(y, n_bands, bandwidth, sr=sr))
                if loglin_feats is not None:
                    # set first band to 0
                    loglin_feats[0] = 0
                    features.append(loglin_feats)
                    feat_names += [f'logpow_band{i}' for i in range(n_bands)]
                else:
                    print(f"ERROR: Log power features returned None")
                    return None, None
                    
            elif ft != 'acoustic-contour-feats':
                print(f"ERROR: Unrecognized feature type: {ft}")
                return None, None
                
        except Exception as e:
            print(f"ERROR: Exception processing feature {ft}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    if len(features) == 0:
        print(f"DEBUG: No mfcc, loglinpower, or rms features extracted")
        return None, None
    
    features = np.concatenate(features)
    return features, feat_names

class ConversationLearner():
    def __init__(self, data_dirs, metadata_fps, data_dirs_test, test_metadata_fps, feats, ags, ags_test, target, test_on_held_out_callers, n_iter, balance_classes, small_class_thresh):
        self.data_dirs = data_dirs
        self.feats = feats
        self.ags = ags
        self.ags_test = ags_test
        self.target = target
        self.n_iter = n_iter
        self.balance_classes = balance_classes
        self.test_on_held_out_callers = test_on_held_out_callers
        self.data_dirs_test = data_dirs_test

        if target=='context':
            remove_dict = {}
        elif target=='caller':
            remove_dict = {'QR2':['B', 'C', 'D']}
        else:
            remove_dict = dict()
        # Add calf exclusion if requested
        if 'AgeCaller' in remove_dict:
            remove_dict['AgeCaller'].append('0B')
        else:
            remove_dict['AgeCaller'] = ['0B']
        if 'AgeClassCaller' in remove_dict:
            remove_dict['AgeClassCaller'].append('0B')
        else:
            remove_dict['AgeClassCaller'] = ['0B']

        if OTHERS_FAMS == 'non_EB':
            non_eb_remove_dict = dict(**remove_dict, Families=['EB'])
        elif OTHERS_FAMS == 'other_EB':
            non_eb_remove_dict = dict(remove_dict, Families=['AA'], CallerName=['Echo', 'Ella', 'Emma', 'Enid', 'Erin', 'Eudora'])
        else:
            assert OTHERS_FAMS == 'all'

        self.metadatas = [filter_metadata_csv(x, non_eb_remove_dict if 'non_EB' in x else None) for x in metadata_fps]
        classes_to_keep = [k for k,v in pd.concat(self.metadatas).ContextTypeNew.value_counts().items() if v > small_class_thresh]
        all_contexts = pd.concat(self.metadatas).ContextTypeNew.value_counts()
        for context, count in all_contexts.items():
            status = "KEPT" if context in classes_to_keep else "EXCLUDED"
            reason = ""
            if count <= small_class_thresh:
                reason = f"(too few: {count} <= {small_class_thresh})"

        if test_metadata_fps == metadata_fps:
            self.test_metadatas = None
        else:
            self.test_metadatas = [filter_metadata_csv(x, non_eb_remove_dict if 'non_EB' in x else None) for x in test_metadata_fps]
            test_classes_to_keep = [k for k,v in pd.concat(self.test_metadatas).ContextTypeNew.value_counts().items() if v > small_class_thresh]
            classes_to_keep = [x for x in classes_to_keep if x in test_classes_to_keep]
            self.test_metadatas = [x[x['ContextTypeNew'].isin(classes_to_keep)] for x in self.test_metadatas]
        self.metadatas = [x[x['ContextTypeNew'].isin(classes_to_keep)] for x in self.metadatas]
        self.imp = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def load_context_data(self, data_dir, metadata, ags):
        
        file_label_pairs = []  # Store (wav_fp, class_name) pairs

        for _, row in metadata.iterrows():
            fname = row['SndFile']
            if fname.endswith('.wav'):
                wav_fp = os.path.join(data_dir, fname)
            else:
                wav_fp = os.path.join(data_dir, f'{fname}.wav')
            if os.path.basename(wav_fp) in aetcs+ooncs:
                continue
            file_label_pairs.append((wav_fp, row.ContextTypeNew))
        
        # Check for duplicates
        wav_fps = [fp for fp, _ in file_label_pairs]
        unique_fps = set(wav_fps)
        if len(wav_fps) != len(unique_fps):
            print(f"WARNING: Found {len(wav_fps) - len(unique_fps)} duplicate file paths!")

            # Option 1: Keep first occurrence of each file
            seen_files = set()
            deduplicated_pairs = []
            for wav_fp, class_name in file_label_pairs:
                if wav_fp not in seen_files:
                    deduplicated_pairs.append((wav_fp, class_name))
                    seen_files.add(wav_fp)
               
            file_label_pairs = deduplicated_pairs


            dedup_classes = [class_name for _, class_name in file_label_pairs]

        wav_fp_list = [fp for fp, _ in file_label_pairs]
        X = extract_conv_dset_features(wav_fp_list, self.feats, ags, recompute=RECOMPUTE, ARGS=ARGS)

        X = pd.DataFrame(X)
        
        
        # Check for empty feature matrix
        if X.empty or X.shape[1] == 0:
            print("ERROR: No features were successfully extracted!")
            print(f"Requested features: {self.feats}")
            print(f"Number of audio files: {len(wav_fp_list)}")
            raise ValueError(f"Feature extraction failed - no features extracted for {self.feats}")

        successful_files = set(X.index)
        print(f"Classes: {Counter(label for _, label in file_label_pairs)}")

        y = [label for fp, label in file_label_pairs if fp in successful_files]


        # Verify alignment
        if len(X) != len(y):
            print(f"ERROR: Still misaligned! Features: {len(X)}, Labels: {len(y)}")
            # truncate to match if needed
            min_len = min(len(X), len(y))
            X = X.iloc[:min_len]
            y = y[:min_len]
            print(f"Emergency truncation to {min_len} samples")

        return X, np.array(y)
        
        
    def transform(self, X):
        # Check for empty feature matrix
        if X.shape[1] == 0:
            print("WARNING: Empty feature matrix passed to transform")
            return X  # Return as-is, will be caught upstream
        
        try:
            return self.imp.fit_transform(self.scaler.fit_transform(X))
        except ValueError as e:
            if "0 feature(s)" in str(e):
                print(f"ERROR: Cannot transform empty feature matrix: {X.shape}")
                raise ValueError(f"Empty feature matrix - check feature extraction: {X.shape}")
            else:
                raise e

    def fit_on_train_test(self, X_train, y_train, X_test, y_test):
        if self.transform(X_train).shape == X_train.shape and self.transform(X_test).shape == X_test.shape:
            X_train = self.transform(X_train)
            X_test = self.transform(X_test)
        else:
            combined = np.concatenate([X_train, X_test], axis=0)
            combined = self.transform(combined)
            X_train = combined[:len(X_train)]
            X_test = combined[len(X_train):]

        clf = LogisticRegression(max_iter=self.n_iter, class_weight='balanced' if self.balance_classes else None, random_state=42)
        clf.fit(X_train, y_train)
        class_feat_importances = clf.coef_
        if class_feat_importances.shape[0]==1:
            class_feat_importances = np.concatenate([class_feat_importances/2, -class_feat_importances/2], axis=0)

        predictions = clf.predict(X_test)
        global_importances = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42).importances_mean
        return predictions, global_importances, class_feat_importances

    def load_data(self, data_dirs, metadatas, ags):
        X_list, y_list, origX_list = [], [], []
        for dd, md in zip(data_dirs, metadatas):
            origX, y = self.load_context_data(dd, md, ags)
            X = origX.values
            X_list.append(X)
            y_list.append(y)
            origX_list.append(origX)
    
        origX_combined = pd.concat(origX_list)
        return origX_combined, np.concatenate(y_list)

    def train(self):
        origX, y = self.load_data(self.data_dirs, self.metadatas, self.ags)
        feature_names = origX.columns
        X = origX.values

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        joined_feats = '_'.join(self.feats)
        if ARGS.exp_name is not None:
            expname = ARGS.exp_name
        else:
            tr_data_name = '+'.join(os.path.basename(x) for x in ARGS.data_dirs)
            ts_data_name = '+'.join(os.path.basename(x) for x in ARGS.data_dirs_test)
            expname = f'tr_{tr_data_name}_{"-".join(self.ags)}-ts_{ts_data_name}_{"-".join(self.ags_test)}-{joined_feats}'
            if ARGS.test_on_held_out_callers:
                expname = expname + '-loocaller'
            if EXCLUDE_CALVES:
                expname = expname + '-nocalves'

        print(f"Experiment name: {expname}")
        os.makedirs(f'context_prediction_output/{expname}', exist_ok=True)

        # Combine metadata for easier access
        combined_metadata = pd.concat(self.metadatas) if self.test_metadatas is None else None
        combined_test_metadata = pd.concat(self.test_metadatas) if self.test_metadatas is not None else None
        

        if self.test_metadatas is None: # do k-fold on self.metadata
            if self.ags_test == self.ags:
                X_test = X
                origX_test = origX
            else:
                origX_test, _ = self.load_data(self.data_dirs, self.metadatas, self.ags_test)
                X_test = origX_test.values
                
                assert not (X_test==X).all()
            if self.test_on_held_out_callers:
               # Combine all metadata and set proper index
                combined_metadata = pd.concat(self.metadatas)
            
                # Reset index to use SndFile (the audio filename column)
                if 'SndFile' in combined_metadata.columns:
                    combined_metadata = combined_metadata.set_index('SndFile')
                else:
                    # If no SndFile column, use the existing index but clean it
                    combined_metadata = combined_metadata.reset_index().set_index('index')
            
                # Extract filenames from successful feature extractions
                audio_file_ids = [os.path.basename(x).removesuffix('.wav') for x in origX.index]
                audio_file_ids = [os.path.basename(x) for x in origX.index]
                
                # Create mapping from successful audio files to their callers
                successful_file_to_caller = {}
                for audio_file_id in audio_file_ids:
                    if audio_file_id in combined_metadata.index:
                        if 'Caller' in combined_metadata.columns:
                            caller = combined_metadata.loc[audio_file_id, 'Caller']
                        elif 'CallerName' in combined_metadata.columns:
                            caller = combined_metadata.loc[audio_file_id, 'CallerName']
                        else:
                            caller = 'Unknown'
                        successful_file_to_caller[audio_file_id] = caller
                
                # Create groups array that matches the order of successful extractions
                groups = []

                for audio_path in origX.index:  # This is in the same order as X and y
                    audio_file_id = os.path.basename(audio_path)
                    if audio_file_id in successful_file_to_caller:
                        caller = successful_file_to_caller[audio_file_id]
                        # Fix: Handle cases where caller might be a list/array
                        if isinstance(caller, (list, np.ndarray, pd.Series)):
                            # Take the first value if it's a sequence
                            caller = caller[0] if len(caller) > 0 else 'Unknown'
                        elif pd.isna(caller):
                            # Handle NaN values
                            caller = 'Unknown'        
                        groups.append(str(caller))  # Ensure it's a string
                    else:
                        groups.append('Unknown')
                
                groups = np.array(groups, dtype=str)  # Explicitly set dtype to string
                
                # don't include unknown callers
                mask = groups != 'Unknown'
                X = X[mask]
                y_enc = y_enc[mask]
                origX = origX.iloc[mask]
                groups = groups[mask]
                
                if len(set(groups)) < 2:
                    print("ERROR: Need at least 2 different callers for LOGO")
                    # Fallback to regular cross-validation
                    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                    folds = kf.split(X, y_enc)

                if 'groups' in locals():
                    logo = LeaveOneGroupOut()
                    folds = logo.split(X, y_enc, groups=groups)
            else:
                # Regular k-fold cross-validation
                kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
                folds = kf.split(X, y_enc)

            all_predictions = np.zeros_like(y_enc)
            all_class_feat_importances = []
            all_global_feat_importances = []
            fold_test_idxs = []
            for (train_idx, test_idx) in folds:
                X_train_fold, X_test_fold= X[train_idx], X_test[test_idx]
                y_train_fold, y_test_fold = y_enc[train_idx], y_enc[test_idx]
                fold_predictions, fold_global_feat_importances, fold_class_feat_importances = self.fit_on_train_test(X_train_fold, y_train_fold, X_test_fold, y_test_fold)
                all_predictions[test_idx] = fold_predictions
                fold_test_idxs.append(test_idx.tolist())

                all_class_feat_importances.append(fold_class_feat_importances)
                all_global_feat_importances.append(fold_global_feat_importances)

            avg_class_feat_importances = np.mean(all_class_feat_importances, axis=0)
            avg_global_feat_importances = np.mean(all_global_feat_importances, axis=0)
            y_test_enc = y_enc
            save_and_plot_results(y_test_enc, all_predictions, le, expname, do_open=SHOW_PLOTS)
            
            save_predictions_to_file(origX, y_test_enc, all_predictions, le, expname, metadata=combined_metadata)
        
            significance_info = {'fold_test_idxs': fold_test_idxs, 'all_preds': all_predictions.tolist(), 'y_test_enc': y_test_enc.tolist()}

        else:
            origX_test, y_test = self.load_data(self.data_dirs_test, self.test_metadatas, self.ags_test)
            X_test = origX_test.values
            
            y_test_enc = le.fit_transform(y_test)
            all_predictions, avg_global_feat_importances, avg_class_feat_importances = self.fit_on_train_test(X, y_enc, X_test, y_test_enc)
            fold_test_idxs = 'didnt use cross-fold'

            save_and_plot_results(y_test_enc, all_predictions, le, expname, do_open=SHOW_PLOTS)
            
            save_predictions_to_file(origX_test, y_test_enc, all_predictions, le, expname, metadata=combined_test_metadata)
        
            significance_info = {'fold_test_idxs': fold_test_idxs, 'all_preds': all_predictions.tolist(), 'y_test_enc': y_test_enc.tolist()}

        os.makedirs(f'context_prediction_output/{expname}', exist_ok=True)

        with open(f'context_prediction_output/{expname}/significance_info.json', 'w') as f:
            json.dump(significance_info, f)

        def readablify(x):
            if 'logpow_band' not in x:
                return x
            for ag in self.ags:
                if x.startswith(ag):
                    x = x.removeprefix(ag+'-')
                    ag_used = ag
                    break
            band_num = int(x.removeprefix('logpow_band'))
            freq_range = f'{10*band_num:g}-{10*(1+band_num):g}'
            return f'{ag_used}_DB_{freq_range}Hz'

        rows = []

        # Filter to only loglinpow features
        loglinpow_indices = [i for i, name in enumerate(feature_names) if 'logpow_band' in name]
        loglinpow_class_importances = avg_class_feat_importances[:, loglinpow_indices]

        feat_imports_df = pd.DataFrame(loglinpow_class_importances, index=le.classes_).T.iloc[::-1]
        feat_imports_df = feat_imports_df.head(30)  # Keep first 30 rows
        feat_imports_df.index = list(reversed([f'{10*i}-{10*(i+1)}' for i in range(len(feat_imports_df))]))
        feat_imports_df.index.name = 'Frequency Band (Hz)'

        feat_imports_df.to_csv(f'context_prediction_output/{expname}/{expname}-all-importances.csv')
        plt.clf()
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(feat_imports_df, cmap="viridis", annot=True, annot_kws={"size":6})
        ax.xaxis.tick_top()
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.savefig(f'context_prediction_output/{expname}/{expname}-all-importances-heatmap.png')
        f = open(f'context_prediction_output/{expname}/{expname}-top-importances.txt', 'w')
        for clf_row_num, class_idx in enumerate(np.unique(y_test_enc)):
            class_name = le.classes_[class_idx]
            top_indices = np.argsort(avg_class_feat_importances[clf_row_num])[::-1]
            if N_FEAT_IMPORTANCES>0:
                print(f'{class_name}:', file=f)
            for imp_idx in top_indices[:N_FEAT_IMPORTANCES]:
                print(f"{readablify(feature_names[imp_idx])}: {avg_class_feat_importances[class_idx, imp_idx]:.4f}", file=f)
            for idx in top_indices:
                rows.append({
                    'feature_name': readablify(feature_names[idx]),
                    'status': class_name,
                    'importance': round(avg_class_feat_importances[clf_row_num][idx], 4)
                })

        # Global permutation importances
        top_global_indices = np.argsort(avg_global_feat_importances)[::-1][:10]
        print('\nGlobal:', file=f)
        for imp_idx in top_global_indices:
            print(f"{readablify(feature_names[imp_idx])}: {avg_global_feat_importances[imp_idx]:.4f}", file=f)
        for idx in top_global_indices:
            rows.append({
                'feature_name': readablify(feature_names[idx]),
                'status': 'global',
                'importance': round(avg_global_feat_importances[idx], 4)
            })

        f.close()

        acc, f1, _, _, _, fake_accs_pred, p_pred, prop_acc, maj_acc = compare_acc_to_chance(all_predictions, y_test_enc, y_enc)

        save_results_to_table(
            experiment_name=expname,
            train_data=str(self.data_dirs),
            test_data=str(self.data_dirs_test),
            features=str(self.feats),
            aggregation=str(self.ags),
            accuracy=acc,
            f1_macro=f1,
            fake_accs_pred=fake_accs_pred,
            p_value_pred_dist=p_pred,
            prop_accuracy=prop_acc,
            majority_accuracy=maj_acc,
            n_samples=len(y_test_enc),
            n_classes=len(np.unique(y_test_enc)),
            results_file=f'context_prediction_output/{expname}/{expname}_results.txt'
        )
    
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['EB', 'nonEB'], required=True,
                    help='Top-level data folder (EB or nonEB)')
    parser.add_argument('--subset', type=str, choices=['single', 'chorus'], required=True,
                    help='Sub-folder within dataset (single or chorus)')
    parser.add_argument('--dataset-test', type=str, choices=['EB', 'nonEB'],
                    help='Test dataset (defaults to --dataset)')
    parser.add_argument('--subset-test', type=str, choices=['single', 'chorus'],
                    help='Test subset (defaults to --subset)')
    parser.add_argument('--test-on-held-out-callers', action='store_true')
    parser.add_argument('--exp-name', type=str, default=None,
                    help='Custom name for experiment results folder (overrides auto-generated name)')

    ARGS = parser.parse_args()

    data_dir = os.path.join(RUMBLES_DIR, ARGS.dataset, ARGS.subset)
    dataset_test = ARGS.dataset_test if ARGS.dataset_test is not None else ARGS.dataset
    subset_test = ARGS.subset_test if ARGS.subset_test is not None else ARGS.subset
    data_dir_test = os.path.join(RUMBLES_DIR, dataset_test, subset_test)

    ARGS.data_dirs = [data_dir]
    ARGS.data_dirs_test = [data_dir_test]
    ARGS.ags_test = AGS

    metadata_fps = [data_dir + '_metadata.csv']
    test_metadata_fps = [data_dir_test + '_metadata.csv']
    tr = ConversationLearner(ARGS.data_dirs, metadata_fps, ARGS.data_dirs_test, test_metadata_fps, FEATS, AGS, ARGS.ags_test, TARGET, test_on_held_out_callers=ARGS.test_on_held_out_callers, n_iter=N_ITER, balance_classes=BALANCE_CLASSES, small_class_thresh=SMALL_CLASS_THRESH)
    tr.train()
