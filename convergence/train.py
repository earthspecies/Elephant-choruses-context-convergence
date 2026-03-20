import os
import csv
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import f1_score
import json
from scipy.signal import resample
from scipy import stats
from natsort import natsorted
import glob
from collections import Counter
import re
import sys
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import mode
from utils import filter_metadata_csv, save_and_plot_results

from scipy.stats import skew
from scipy.fft import fft
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


def forest_feat_importances(rf):
    """importances[i,j] is the mean decrease in impurity in class i across all nodes that split by feature j"""
    n_classes = rf.n_classes_
    n_features = rf.n_features_in_
    importances = np.zeros((n_classes, n_features))

    for tree in rf.estimators_:
        tree_imp = np.zeros((n_classes, n_features))
        for node in range(tree.tree_.node_count):
            if tree.tree_.children_left[node] != tree.tree_.children_right[node]:
                feature = tree.tree_.feature[node]
                impurity = tree.tree_.impurity[node]
                left = tree.tree_.children_left[node]
                right = tree.tree_.children_right[node]
                n_node_samples = tree.tree_.n_node_samples[node]
                n_left = tree.tree_.n_node_samples[left]
                n_right = tree.tree_.n_node_samples[right]
                w_left = n_left / n_node_samples
                w_right = n_right / n_node_samples
                impurity_left = tree.tree_.impurity[left]
                impurity_right = tree.tree_.impurity[right]
                for c in range(n_classes):
                    p_node = tree.tree_.value[node][0, c] / n_node_samples
                    p_left = tree.tree_.value[left][0, c] / n_left
                    p_right = tree.tree_.value[right][0, c] / n_right
                    class_gain = p_node * impurity - (w_left * p_left * impurity_left + w_right * p_right * impurity_right)
                    tree_imp[c, feature] += class_gain
        importances += tree_imp

    importances /= len(rf.estimators_)
    return importances

# SCK using different file here
#with open('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/less512_wavs.txt') as f:
    near_emptys = f.read().split('\n')

with open('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/empty_wavs_louis.txt') as f:
    near_emptys = f.read().split('\n')


with open('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/only_one_nonempty_csvs_louis.txt') as f:
    ooncs = f.read().split('\n')

with open('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/alone_empty_time_csvs.txt') as f:
    aetcs = f.read().split('\n')

conv_level_feat_names = ['n-calls', 'overlap-ratio', 'latency']

def resample_contour(contour, target_len):
    x_old = np.linspace(0, 1, len(contour))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, contour)

def compute_log_power_in_bands(x, n_freq, bandwidth, sr=None, ARGS=None):
    
    if isinstance(x, str):
        assert sr is None
        y_sp, sr = librosa.load(x, sr=None)
    else:
        assert isinstance(x, np.ndarray)
        assert sr is not None
        y_sp = x
    assert sr==16000
    n_fft = 800
    if len(y_sp) < n_fft:
        return None
    win_length = 512
    hop = 128
    hz_per_row = sr / n_fft
    rows_per_band = int(bandwidth / hz_per_row)
    S = np.abs(librosa.stft(y_sp, n_fft=n_fft, hop_length=hop, win_length=win_length))**2
    S_db = librosa.power_to_db(S, ref=np.max)
    #FOLLOWING LINES ARE FOR OMITTING <12.5Hz
    #S_db[:int(100/hz_per_row), :] = S_db.min()
    #print("replacing <12.5Hz with min db value: ", S_db.min())
    #plt.figure(figsize=(10, 4))
    #librosa.display.specshow(S_db, sr=sr, hop_length=hop, x_axis='time', y_axis='log')
    #plt.colorbar(format='%+2.0f dB')
    #plt.title('Power Spectrogram')
    #plt.tight_layout()
    #plt.savefig('/tmp/spec.png')
    #os.system('/usr/bin/xdg-open /tmp/spec.png')

    grouped_db = S_db[:n_freq*rows_per_band].reshape(n_freq, rows_per_band, -1)
    spf = grouped_db.mean(axis=(1,2))
    #print(f"compute_log_power_in_bands: n_freq={n_freq}, spf.shape={spf.shape}, spf.ndim={spf.ndim}")
    if spf.ndim == 0:
        print(f"  WARNING: spf is 0-D! grouped_db.shape={grouped_db.shape}")
        breakpoint()
    # SCK omit lowest band
    if ARGS is not None and ARGS.ignore_first_band:
        spf[0] = S_db.min()
    # print("Log power in bands:", spf)
    return spf

def db_mel_spec_from_fp(fp):#, recompute=True):
    cache_fp = fp.replace('.wav', '_powerbands.npy')
    y, sr = librosa.load(fp, sr=None)
    assert sr == 16000
    if len(y) < 512:
        breakpoint()
    n_fft = int(sr * 0.350/8); hop_length = int(n_fft * 0.1)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hamming', n_mels=13, fmin=0, fmax=2000, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    np.save(cache_fp, S_db)
    return S_db

def convergence_rate(seq, use_rel_rank=True, filenames=None):
    diffs = np.abs(np.diff(seq))
    diffs = diffs[diffs > 0]  # avoid log(0)
    if len(diffs) < 2:
        return np.nan
    x = np.arange(len(diffs))
    y = np.log(diffs)
    slope, _ = np.polyfit(x, y, 1)
    
    if use_rel_rank:
        slope *= len(diffs) # adjust as if using relative rank
    #if slope==0:
    #    breakpoint()
    #print('conv diffs:', diffs)
    if filenames is not None:
        return -slope, diffs, filenames
    return -slope  # higher means faster False


def euclidean_convergence_rate(seq, use_rel_rank=True, filenames=None, starts=None, ends=None):
    if not isinstance(seq, np.ndarray):
        seq = np.array(seq)
    if seq.ndim==1:
        seq = np.expand_dims(seq, 1)
    diffs = (np.diff(seq, axis=0)**2).sum(axis=1)**0.5
    diffs = diffs[diffs > 0]  # avoid log(0)
    if len(diffs) < 2:
        return np.nan
        # return 0
    x = np.arange(len(diffs))
    y = np.log(diffs)
    slope, _ = np.polyfit(x, y, 1)
    if use_rel_rank:
        slope *= len(diffs) # adjust as if using relative rank        
    #if slope==0:
    #    breakpoint()
    #print('euc conv diffs:', diffs)
    if filenames is not None:
        return -slope, diffs, filenames, starts, ends
    return -slope # higher means faster False

def aggregate_call_feats_across_conv(features_by_call, feat_names, ag_methods, filenames=None, starts=None, ends=None):
    agged_features_by_name = {}
    if 'mean' in ag_methods:
        mean_feats = features_by_call.mean(axis=0)
        for fn, fv in zip(feat_names, mean_feats):
            agged_features_by_name[f'mean-{fn}'] = fv
    if 'std' in ag_methods:
        std_feats = features_by_call.std(axis=0)
        for fn, fv in zip(feat_names, std_feats):
            agged_features_by_name[f'std-{fn}'] = fv
    if 'cnvrg' in ag_methods and len(features_by_call) > 2:
        cnvrg_feats = np.apply_along_axis(convergence_rate, 0, features_by_call)
        for fn, fv in zip(feat_names, cnvrg_feats):
            agged_features_by_name[f'cnvrg-{fn}'] = fv
        if (cnvrg_feats[1:]==0).any():
            breakpoint()
        agged_features_by_name['overall-cnvrg'] = euclidean_convergence_rate(features_by_call, filenames=filenames, starts=starts, ends=ends)


        
    if 'first' in ag_methods:
        for fn, fv in zip(feat_names, features_by_call[0]):
            agged_features_by_name[f'first-{fn}'] = fv
    if 'last' in ag_methods:
        for fn, fv in zip(feat_names, features_by_call[-1]):
            agged_features_by_name[f'last-{fn}'] = fv
    return agged_features_by_name

def acoustic_features_from_contours(X, n_pcs=10, n_fft_peaks=4):
    N, nf, nt = X.shape
    pca_input = X.mT.reshape(-1, nf)

    scaler = RobustScaler()
    pca_input = scaler.fit_transform(pca_input)

    pca = PCA(n_components=n_pcs)
    pca_output = pca.fit_transform(pca_input)  # (N*nt, n_pcs)
    X_pca = pca_output.reshape(N, nt, n_pcs).mT # (N, n_pcs, nt)

    median = np.median(X_pca, axis=2)
    skewness = skew(X_pca, axis=2)
    min_extent = np.min(X_pca, axis=2)
    stat_extent = np.percentile(X_pca, 75, axis=2) - np.percentile(X_pca, 25, axis=2)

    stat_feats = np.stack((median, skewness, min_extent, stat_extent), axis=2) # (N, n_pcs, 4)

    freqs = np.fft.fftfreq(nt)
    freq_domain_pcs = np.abs(fft(X_pca, axis=2))
    freq_domain_peaks = np.argpartition(-freq_domain_pcs, n_fft_peaks, axis=2)[:, :, :n_fft_peaks]
    freq_domain_peak_vals = np.take_along_axis(freq_domain_pcs, freq_domain_peaks, axis=2)
    freq_domain_peak_freqs = np.array([[freqs[row] for row in mat] for mat in freq_domain_peaks])
    fft_feats = np.concatenate([freq_domain_peak_freqs, freq_domain_peak_vals], axis=2) # (N, n_pcs, 2*n_fft_peaks)
    pca_feats = np.concatenate([stat_feats, fft_feats], axis=2)
    pca_feat_names = ['median', 'skewness', 'min', 'iqr'] + [f'fft_freq_{i+1}' for i in range(n_fft_peaks)] + [f'fft_mag_{i+1}' for i in range(n_fft_peaks)]
    dpoint_feats = pca_feats.reshape(N, n_pcs*(4 + 2*n_fft_peaks))
    dpoint_feat_names = [f'pca{i}_{featname}' for featname in pca_feat_names for i in range(n_pcs)]
    dpoint_feats_df = pd.DataFrame(dpoint_feats, columns=dpoint_feat_names)
    return dpoint_feats_df

def extract_conv_features(audio_fp, feats, ags, enforce_no_overlaps=False, recompute=True):
    fn_base = os.path.basename(audio_fp).removesuffix('.wav')
    data_dir = os.path.dirname(audio_fp)
    filenames = []
    if enforce_no_overlaps and '-noverlaps' not in fn_base:
        fn_base += '-noverlaps'
    
    feats_by_call = []
    i = 0
    conv_feats = [x for x in feats if x in conv_level_feat_names]
    call_feats = [x for x in feats if x not in conv_feats]
    starts = []
    ends = []
    if ags == ['single']:
        assert len(conv_feats) == 0, 'cant have conv feats if treating as single'
        single_split_feats, call_feat_names = extract_call_features(audio_fp, call_feats, recompute=recompute)
        features_by_name = {f'single-{k}':v for k,v in zip(call_feat_names, single_split_feats)}
        return features_by_name

    while True:
        i += 1
        
        if not os.path.exists(os.path.join(data_dir, f'{fn_base}_{i}.wav')):
            break
        fn = f'{fn_base}_{i}.wav'
        

        #SCK: make sure not empty before we proceed
        if fn in near_emptys:
            #print('skipping near empty call', fn)
            continue

        y, sr = librosa.load(os.path.join(data_dir, fn), sr=None)
        assert sr == 16000
        if len(y) < 512:
            print('skipping empty call', fn)
            continue
        # end SCK
        filenames.append(fn)

        # SCK fixed to open correct csv
        if enforce_no_overlaps:
            temp = os.path.join(data_dir, fn.replace('-noverlaps','')).replace('.wav', '.csv')
            times = pd.read_csv(temp)
        else:
            times = pd.read_csv(os.path.join(data_dir, fn).replace('.wav', '.csv'))
        starts.append(times.loc[0, 'time'])
        ends.append(times['time'].iloc[-1])
        
        
        ('processing call', fn)
        #print(' start time:', starts[-1], 'end time:', ends[-1])
        this_split_feats, call_feat_names = extract_call_features(os.path.join(data_dir, fn), call_feats, recompute=recompute)
        
        if this_split_feats is not None:
            feats_by_call.append(this_split_feats)
        else:
            print('this_split_feats is None for ', fn)
            breakpoint()
    feats_by_call = np.array(feats_by_call)
    starts = np.array(starts)
    ends = np.array(ends)
    
    if feats_by_call.size==0:
        if not ( any(fn_base in x for x in near_emptys)):
            print('problem with features')
            breakpoint()
        #('len features is 0', fn_base)
        return None, None
    if ags == 'none':
        return feats_by_call
    features_by_name = aggregate_call_feats_across_conv(feats_by_call, call_feat_names, ags, filenames, starts, ends)
    if 'n-calls' in conv_feats:
        features_by_name['n_calls'] = i-1
    if 'overlap-ratio' in conv_feats and len(starts) > 1:
        overlap_time = np.maximum(0, starts[1:] - ends[:-1]).sum()
        total_time = (ends - starts).sum()
        features_by_name['overlap-ratio'] = overlap_time / total_time
    if 'latency' in conv_feats and len(starts) > 1:
        intercall_time = np.maximum(0, ends[:-1] - starts[1:]).sum()
        features_by_name['latency'] = intercall_time.mean()
        if np.isnan(features_by_name['latency']):
            breakpoint()

    return features_by_name

def extract_conv_dset_features(audio_fp_list, feats, ag_methods, enforce_no_overlaps=False, recompute=False):
    indiv_convs_feats = [x for x in feats if x not in ['acoustic-contour-feats']]
    if len(indiv_convs_feats) == 0:
        features_by_fp = pd.DataFrame(index=audio_fp_list)
    else:
        features_by_fp = {}
        for afp in audio_fp_list:
            features_by_fp[afp] = extract_conv_features(afp, indiv_convs_feats, ag_methods, recompute=recompute)
        features_by_fp = pd.DataFrame(features_by_fp).T
    if 'acoustic-contour-feats' in feats:
        ac_feats = extract_acoustic_contour_conv_features(audio_fp_list, ag_methods)
        features_by_fp = pd.concat([features_by_fp, ac_feats], axis=1)
    return features_by_fp

def extract_acoustic_contour_call_features(audio_fp_list):
    all_db_mel_specs_dict = {afp:db_mel_spec_from_fp(afp) for afp in audio_fp_list if afp not in near_emptys}
    target_len = int(np.median([x.shape[1] for x in all_db_mel_specs_dict.values()]))
    all_contours = np.stack([resample(x, target_len, axis=1) for x in all_db_mel_specs_dict.values()])
    ac_feats = acoustic_features_from_contours(all_contours)
    ac_feats.index = all_db_mel_specs_dict.keys()
    return ac_feats

def extract_acoustic_contour_conv_features(audio_fp_list, ag_methods, level='conv'):
    n_calls_per_conv = {}
    all_db_mel_specs_dict = {}
    for afp in audio_fp_list:
        data_dir = os.path.dirname(afp)
        i = 0
        nc = 0

        fn_base = os.path.basename(afp).removesuffix('.wav')
        while True:
            i += 1
            fn = f'{fn_base}_{i}.wav'
            if fn in near_emptys:
                continue
            if not os.path.exists(call_fp:=os.path.join(data_dir, fn)):
                break
            S_db = db_mel_spec_from_fp(call_fp)
            if len(S_db) != 0:
                all_db_mel_specs_dict[fn] = S_db
                nc += 1

        n_calls_per_conv[afp] = nc
    target_len = int(np.median([x.shape[1] for x in all_db_mel_specs_dict.values()]))
    all_contours = np.stack([resample(x, target_len, axis=1) for x in all_db_mel_specs_dict.values()])
    all_pardo_feats = acoustic_features_from_contours(all_contours)
    conv_split_points = np.cumsum(list(n_calls_per_conv.values()))
    starts = [0]+list(conv_split_points)[:-1]
    dset_level_feats = [aggregate_call_feats_across_conv(all_pardo_feats.values[i:j], feat_names=all_pardo_feats.columns, ag_methods=ag_methods) for i,j in zip(starts, conv_split_points)]
    dset_level_feats_by_conv = pd.DataFrame(dset_level_feats, index=n_calls_per_conv.keys())
    return dset_level_feats_by_conv

def extract_call_features(audio_fp, feats, frame_mfcc_std=False, recompute=False, ARGS=None):
    y, sr = librosa.load(audio_fp, sr=None)
    
    if len(y)==0:
        return None, None
    
    scale_by = 2**(16-1) * 0.7 / y.max()
    y *= scale_by
    y -= y.mean()
    fn_base = os.path.basename(audio_fp).removesuffix('.wav')
    data_dir = os.path.dirname(audio_fp)
    os.makedirs(cache_dir:=f'cached_feats/{fn_base}', exist_ok=True)
    def load_or_save(cache_fp, compute_fn):
        if os.path.exists(cache_fp) and not recompute:
            # print('loading from cache', cache_fp)
            return np.load(cache_fp)
        # print('computing and saving to cache', cache_fp)
        result = compute_fn()
        np.save(cache_fp, result)
        return result

    features = []
    feat_names = []
    for ft in feats:
        cache_fp = os.path.join(cache_dir, f'{ft}.npy')
        if ft.startswith('mfcc'):
            n_mfcc = int(ft.split('-')[1])
            mfcc_feats = load_or_save(cache_fp, lambda: np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=512, fmin=100), axis=1))
            features.append(mfcc_feats)
            feat_names += [f'mfcc{i}' for i in range(n_mfcc)]
        elif ft.startswith('loglinpow'):
            n_bands = int(ft.split('-')[1])
            bandwidth = int(ft.split('-')[2])
            assert n_bands <= 50 # make sure didn't mix the two up
            loglin_feats = load_or_save(cache_fp, lambda: compute_log_power_in_bands(y, n_bands, bandwidth, sr=sr, ARGS=ARGS))
            if loglin_feats is None:
                print(f"Warning: compute_log_power_in_bands returned None for {audio_fp}")
                return None, None

            features.append(loglin_feats)
            feat_names += [f'logpow_band{i}' for i in range(n_bands)]
        elif ft=='rms':
            rms_cache_fp = os.path.join(cache_dir, 'rms.npy')
            rms = load_or_save(rms_cache_fp, lambda: (y**2).mean(keepdims=True)**0.5)
            features.append(rms)
            feat_names.append('rms')
        elif 'poly' in feats:
            poly_params_fp = os.path.join(data_dir, f'{fn_base}.poly3_params.npy')
            if not os.path.exists(poly_params_fp):
                print('no poly at', poly_params_fp)
                return None
            pp = np.load(poly_params_fp)
            x = abs(pp).max()
            clip_val = 10000
            if x >= clip_val:
                print(f'absurdly extreme values in {poly_params_fp}, clipping')
                pp = np.clip(pp, a_min=-clip_val, a_max=clip_val)
            features.append(pp)
            feat_names += [f'poly_{i}' for i in range(4)]
        elif ft.startswith('formants'):
            n_formants = int(ft.split('-')[1])
            formants_fp = os.path.join(cache_dir, 'formants.npy')
            if not os.path.exists(formants_fp):
                print('no formants at', formants_fp)
                return None
            fmts = np.load(formants_fp)[:n_formants]
            features.append(fmts)
            feat_names += [f'formant_{i}' for i in range(n_formants)]
        elif ft == 'median-pitch':
            freq_df = pd.read_csv(os.path.join(data_dir, f'{fn_base}.csv'))
            pitch = np.log(np.median(freq_df.frequency, keepdims=True))
            features.append(pitch)
            feat_names.append('median-pitch')
        else:
            sys.exit(f'unrecognised feature: {ft}')
    # debug
    
    features = np.concatenate(features)
    return features, feat_names

class ConversationLearner():
    def __init__(self, data_dirs, metadata_fps, data_dirs_test, test_metadata_fps, feats, ags, ags_test, target, split_by, test_on_held_out_callers, ml_model, n_iter, balance_classes, n_rank_bins, small_class_thresh, kfold_within_train):
        self.data_dirs = data_dirs
        self.metadata_fps = metadata_fps
        self.feats = feats
        self.ags = ags
        self.ags_test = ags_test
        self.target = target
        self.split_by = split_by
        self.ml_model = ml_model
        self.n_iter = n_iter
        self.balance_classes = balance_classes
        self.test_on_held_out_callers = test_on_held_out_callers
        self.n_rank_bins = n_rank_bins
        self.data_dirs_test = data_dirs_test
        self.kfold_within_train = kfold_within_train

        if target=='context':
            remove_dict = {'QR1':['B', 'C', 'D']}
        elif target=='caller':
            remove_dict = {'QR2':['B', 'C', 'D']}
        else:
            remove_dict = dict()
        if ARGS.others_fams == 'non_EB':
            non_eb_remove_dict = dict(**remove_dict, Families=['EB'])
        elif ARGS.others_fams == 'other_EB':
            non_eb_remove_dict = dict(remove_dict, Families=['AA'], CallerName=['Echo', 'Ella', 'Emma', 'Enid', 'Erin', 'Eudora'])
        else:
            assert ARGS.others_fams == 'all'

        self.metadatas = [filter_metadata_csv(x, non_eb_remove_dict if 'non_EB' in x else None) for x in metadata_fps]
        classes_to_keep = [k for k,v in pd.concat(self.metadatas).ContextType.value_counts().items() if v > small_class_thresh and k not in ARGS.exclude_classes]
        if test_metadata_fps == metadata_fps:
            self.test_metadatas = None
        else:
            self.test_metadatas = [filter_metadata_csv(x, non_eb_remove_dict if 'non_EB' in x else None) for x in test_metadata_fps]
            test_classes_to_keep = [k for k,v in pd.concat(self.test_metadatas).ContextType.value_counts().items() if v > small_class_thresh]
            classes_to_keep = [x for x in classes_to_keep if x in test_classes_to_keep]
            self.test_metadatas = [x[x['ContextType'].isin(classes_to_keep)] for x in self.test_metadatas]
        self.metadatas = [x[x['ContextType'].isin(classes_to_keep)] for x in self.metadatas]
        self.imp = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

    def load_context_data(self, data_dir, metadata, ags):
        wav_fp_list = []
        y = []
        assert self.target in ['context', 'caller', 'family']
        for idx, row in metadata.iterrows():
            if row['n_ge512_rumbles'] == 0:
                continue
            wav_fp = os.path.join(data_dir, f'{idx}.wav')
            if os.path.basename(wav_fp) in aetcs+ooncs:
                continue
            if self.target=='context':
                class_name = row.ContextType
            elif self.target=='caller':
                class_name = row.Caller
            else:
                if 'Families' in row.keys() and isinstance(row.Families, str):
                    isEB = 'EB' in row.Families
                elif 'Caller' in row.keys():
                    isEB = row.Caller in ['Echo', 'Ella', 'Emma', 'Enid', 'Erin', 'Eudora']
                else:
                    isEB = row.CallerName in ['Echo', 'Ella', 'Emma', 'Enid', 'Erin', 'Eudora']
                class_name = 'EB' if isEB else 'non-EB'
                print(class_name)
            wav_fp_list.append(wav_fp)
            y.append(class_name)

        X = extract_conv_dset_features(wav_fp_list, self.feats, ags, recompute=ARGS.recompute)
        X = pd.DataFrame(X)
        y = np.array(y)
        return X, y

    def load_call_split_data(self, metadata):
        assert self.split_by=='call'
        X, y_call, y_conv = [], [], []
        self.callidx2convidx = []
        all_call_wav_fps = glob.glob(f'/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/with_others/*_[0-9]*.wav')
        ridx = 0
        for idx, row in metadata.iterrows():
            call_wav_fps = [x for x in all_call_wav_fps if idx in x]
            call_idxs = []
            for fwfp in call_wav_fps:
                if os.path.basename(fwfp) in ooncs:
                    breakpoint()
                    continue
                features, feature_names = extract_call_features(fwfp, self.feats, recompute=ARGS.recompute)

                if features is not None:
                    X.append(features)
                    y_call.append(row.ContextType)
                    call_idxs.append(ridx)
                    ridx += 1
            if len(call_idxs) > 0:
                self.callidx2convidx.append(np.array(call_idxs))
                y_conv.append(row.ContextType)
        return X, y_call, feature_names, y_conv

    def load_rank_data(self, data_dir, metadata, ags, relative_rank):
        assert all(x not in self.feats for x in conv_level_feat_names), 'Cant use these features when predicting rank'
        call_feats = [x for x in self.feats if x != 'acoustic-contour-feats']
        all_files = [x for x in os.listdir(data_dir) if bool(re.search(r'_\d{1,2}.wav', x)) and x not in near_emptys]
        wav_fp_list = []
        assert not any('noverlaps' in x for x in all_files)
        X = {}
        y = []
        for idx, row in metadata.iterrows():
            conv_files = natsorted([x for x in all_files if idx in x])
            if len(conv_files) <= 1:
                continue
            max_cidx = max(int(x.split('_')[-1].removesuffix('.wav')) for x in conv_files) # not just len(conv_files) because some idxs missing
            if max_cidx == 1:
                continue
            for fn in conv_files:
                rank_label = int(fn.split('_')[-1].removesuffix('.wav')) - 1
                if rank_label > 10:
                    continue
                if relative_rank:
                    rank_label = int(self.n_rank_bins*rank_label/max_cidx)
                    if not ( rank_label in [0,1,2]):
                        breakpoint()
                fp = os.path.join(data_dir, fn)
                feats, feature_names = extract_call_features(fp, call_feats, recompute=ARGS.recompute)
                X[fp] = dict(zip(feature_names, feats))
                wav_fp_list.append(fp)
                y.append(rank_label)
            if rank_label==0:
                breakpoint()
        X = pd.DataFrame(X).T
        if 'acoustic-contour-feats' in self.feats:
            X_contour = extract_acoustic_contour_call_features(wav_fp_list)
            X = pd.concat([X, X_contour], axis=1)
        label_counts = Counter(y)
        mask = np.array([label_counts[label] > 5 for label in y])
        y = np.array([str(z) for z in y])
        X_filtered = X.loc[mask]
        y_filtered = y[mask]

        return X_filtered, y_filtered

    def transform(self, X):
        return self.imp.fit_transform(self.scaler.fit_transform(X))

    def fit_on_train_test(self, X_train, y_train, X_test, y_test):
        if self.transform(X_train).shape == X_train.shape and self.transform(X_test).shape == X_test.shape:
            X_train = self.transform(X_train)
            X_test = self.transform(X_test)
        else:
            combined = np.concatenate([X_train, X_test], axis=0)
            combined = self.transform(combined)
            X_train = combined[:len(X_train)]
            X_test = combined[len(X_train):]

        if self.ml_model == 'lr':
            clf = LogisticRegression(max_iter=self.n_iter, class_weight='balanced' if self.balance_classes else None)
            clf.fit(X_train, y_train)
            class_feat_importances = clf.coef_
            if class_feat_importances.shape[0]==1: # sklearn automatically uses sigmoid with n-weight-rows==n-class - 1
                class_feat_importances = np.concatenate([class_feat_importances/2, -class_feat_importances/2], axis=0)
        else:
            clf = AdaBoostClassifier(n_estimators=100)
            clf.fit(X_train, y_train)
            class_feat_importances = forest_feat_importances(clf)

        predictions = clf.predict(X_test)
        global_importances = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=np.random.randint(1)).importances_mean
        return predictions, global_importances, class_feat_importances

    def load_data(self, data_dirs, metadatas, ags):
        X_list, y_list = [], []
        for dd, md in zip(data_dirs, metadatas):
            if self.target in ['context', 'caller', 'family']:
                if self.split_by == 'conv':
                    X, y = self.load_context_data(dd, md, ags)
                else:
                    X, y, feature_names, y_conv = self.load_call_split_data(dd, md)
            elif self.target == 'rank':
                X, y = self.load_rank_data(dd, md, ags, relative_rank=False)
            else:
                assert self.target == 'rel-rank'
                X, y = self.load_rank_data(dd, md, relative_rank=True)
            X_list.append(X); y_list.append(y)
        return pd.concat(X_list), np.concatenate(y_list)

    def train(self):
        origX, y = self.load_data(self.data_dirs, self.metadatas, self.ags)
        feature_names = origX.columns
        X = origX.values
        print(X.shape)

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        joined_feats = '_'.join(self.feats)
        shorten_name_dict = {
                'cleaned_shifted_8-with_others': 'coreEBwo',
                'cleaned_shifted_8-alone': 'coreEBa',
                'non_EB_core-single': 'ncoreEBa' if ARGS.others_fams=='other_EB' else 'nEBa',
                'non_EB_core-multiple': 'ncoreEBwo' if ARGS.others_fams=='other_EB' else 'nEBwo',
                }
        tr_data_names = [x.removeprefix('joyce_poole_elephants/').replace('/','-') for x in ARGS.data_dirs]
        ts_data_names = [x.removeprefix('joyce_poole_elephants/').replace('/','-') for x in ARGS.data_dirs_test]
        tr_data_name = '+'.join(shorten_name_dict[x] for x in tr_data_names)
        ts_data_name = '+'.join(shorten_name_dict[x] for x in ts_data_names)
        expname = f'tr_{tr_data_name}_{"-".join(self.ags)}-ts_{ts_data_name}_{"-".join(self.ags_test)}-{joined_feats}'
        if ARGS.target != 'context':
            expname = expname + f'-{ARGS.target}'
        if ARGS.test_on_held_out_callers:
            expname = expname + '-loocaller'
        print(expname)
        os.makedirs(f'LR_results/{expname}', exist_ok=True)

        if self.kfold_within_train: # do k-fold on self.metadata
            X_test, y_test = self.load_data(self.data_dirs_test, self.test_metadatas, self.ags_test)
            assert (X_test.columns == feature_names).all()
            X_test = X_test.values
            if ARGS.exclude_novel_test:
                mask = (np.expand_dims(y_test,1) == np.unique(y)).any(axis=1)
                X_test = X_test[mask]; y_test = y_test[mask]
            y_test_enc = le.fit_transform(y_test)
            kf = KFold(n_splits=ARGS.n_folds, shuffle=True)
            all_train_predictions = np.zeros_like(y_enc)
            all_test_predictions = np.zeros_like(y_test_enc)
            all_class_feat_importances = []
            all_global_feat_importances = []
            train_folds = kf.split(X, y_enc, groups=None)
            test_folds = kf.split(X_test, y_test_enc, groups=None)
            for (train_train_idx, train_test_idx), (test_train_idx, test_test_idx) in zip(train_folds, test_folds):
                X_train_train_fold, X_train_test_fold, X_test_test_fold = X[train_train_idx], X[train_test_idx], X_test[test_test_idx]
                X_test_fold = np.concatenate([X_train_test_fold, X_test_test_fold], axis=0)
                y_train_train_fold, y_train_test_fold, y_test_test_fold = y_enc[train_train_idx], y_enc[train_test_idx], y_test_enc[test_test_idx]
                y_test_fold = np.concatenate([y_train_test_fold, y_test_test_fold], axis=0)
                fold_predictions, fold_global_feat_importances, fold_class_feat_importances = self.fit_on_train_test(X_train_train_fold, y_train_train_fold, X_test_fold, y_test_fold)
                all_train_predictions[train_test_idx] = fold_predictions[:len(y_train_test_fold)]
                all_test_predictions[test_test_idx] = fold_predictions[len(y_train_test_fold):]

                all_class_feat_importances.append(fold_class_feat_importances)
                all_global_feat_importances.append(fold_global_feat_importances)

            avg_class_feat_importances = np.mean(all_class_feat_importances, axis=0)
            avg_global_feat_importances = np.mean(all_global_feat_importances, axis=0)
            fold_test_idxs = 'used complicated splits with train_train/train_test etc'
            save_and_plot_results(y_enc, all_train_predictions, le, expname, do_open=ARGS.show_plots, suffix=' Train')
            save_and_plot_results(y_test_enc, all_test_predictions, le, expname, do_open=ARGS.show_plots, suffix=' Test')
            significance_info = {'fold_test_idxs': fold_test_idxs, 'all_train_preds': all_train_predictions.tolist(), 'all_test_preds': all_test_predictions.tolist(), 'y_test_enc': y_test_enc.tolist()}

        elif self.test_metadatas is None: # do k-fold on self.metadata
            if self.ags_test == self.ags:
                X_test = X
            else:
                X_test, _ = self.load_data(self.data_dirs, self.metadatas, self.ags_test)
                X_test = X_test.values
                if not ( X_test.shape == X.shape):
                    breakpoint()
                assert not (X_test==X).all()
            if self.test_on_held_out_callers:
                metadata_not_discarded = self.metadata.loc[[os.path.basename(x).removesuffix('.wav') for x in origX.index]]
                groups = metadata_not_discarded['Caller'].values
                logo = LeaveOneGroupOut()
                folds = logo.split(X, y_enc, groups=groups)
            else:
                kf = KFold(n_splits=ARGS.n_folds, shuffle=True, random_state=42)
                folds = kf.split(X, y_enc)

            all_predictions = np.zeros_like(y_enc)
            all_class_feat_importances = []
            all_global_feat_importances = []
            fold_test_idxs = []
            for fold_idx, (train_idx, test_idx) in enumerate(folds):
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
            save_and_plot_results(y_test_enc, all_predictions, le, expname, do_open=ARGS.show_plots)
            significance_info = {'fold_test_idxs': fold_test_idxs, 'all_preds': all_predictions.tolist(), 'y_test_enc': y_test_enc.tolist()}

        else:
            X_test, y_test = self.load_data(self.data_dirs_test, self.test_metadatas, self.ags_test)
            breakpoint()
            if ARGS.exclude_novel_test:
                mask = (np.expand_dims(y_test,1) == np.unique(y)).any(axis=1)
                X_test = X_test[mask]; y_test = y_test[mask]
            y_test_enc = le.fit_transform(y_test)
            all_predictions, avg_global_feat_importances, avg_class_feat_importances = self.fit_on_train_test(X, y_enc, X_test, y_test_enc)
            fold_test_idxs = 'didnt use cross-fold'

            save_and_plot_results(y_test_enc, all_predictions, le, expname, do_open=ARGS.show_plots)
            significance_info = {'fold_test_idxs': fold_test_idxs, 'all_preds': all_predictions.tolist(), 'y_test_enc': y_test_enc.tolist()}

        os.makedirs(f'LR_results/{expname}', exist_ok=True)

        with open(f'LR_results/{expname}/significance_info.json', 'w') as f:
            json.dump(significance_info, f)

        if self.split_by == 'call':
            mean_conv_preds = [mode(all_predictions[x], keepdims=True).mode[0] for x in self.callidx2convidx]
            y_enc = le.fit_transform(y)
            save_and_plot_results(y_enc, mean_conv_preds, le, expname, "conv")

        def readablify(x):
            if 'logpow_band' not in x:
                return x
            for ag in self.ags:
                if x.startswith(ag):
                    x = x.removeprefix(ag+'-')
                    ag_used = ag
                    break
            band_num = int(x.removeprefix('logpow_band'))
            freq_range = f'{12.5*band_num:g}-{12.5*(1+band_num):g}'
            return f'{ag_used}_DB_{freq_range}Hz'

        rows = []

        feat_imports_df = pd.DataFrame(avg_class_feat_importances, index=le.classes_).T.iloc[::-1]
        feat_imports_df.index = reversed([f'{12.5*i}-{12.5*(i+1)}' for i in range(len(feat_imports_df))])
        feat_imports_df.index.name = 'Frequency Band (Hz)'
        feat_imports_df.to_csv(f'LR_results/{expname}/{expname}-all-importances.csv')
        plt.clf()
        plt.figure(figsize=(12, 8))
        ax = sns.heatmap(feat_imports_df, cmap="viridis", annot=True, annot_kws={"size":6})
        ax.xaxis.tick_top()
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.savefig(f'LR_results/{expname}/{expname}-all-importances-heatmap.png')
        f = open(f'LR_results/{expname}/{expname}-top-importances.txt', 'w')
        for clf_row_num, class_idx in enumerate(np.unique(y_test_enc)):
            class_name = le.classes_[class_idx]
            top_indices = np.argsort(avg_class_feat_importances[clf_row_num])[::-1]
            if ARGS.n_feat_importances>0:
                print(f'{class_name}:', file=f)
            for imp_idx in top_indices[:ARGS.n_feat_importances]:
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

        if self.kfold_within_train:
            print('\nTrain:')
            compare_acc_to_chance(all_train_predictions, y_enc, y_enc)
            print('\nTest:')
            compare_acc_to_chance(all_test_predictions, y_test_enc, y_enc)
        else:
            compare_acc_to_chance(all_predictions, y_test_enc, y_enc)

def compare_acc_to_chance(preds, gts_test, gts_train):
    accuracy = (preds==gts_test).mean()
    majority_class_accuracy = np.max(np.bincount(gts_train)) / len(gts_train)
    class_proportions = np.bincount(gts_train) / len(gts_train)
    prop_accuracy = np.sum(class_proportions ** 2)  # Expected accuracy when sampling by class frequency
    print(f'Accuracy: {accuracy:.4f}')
    print('\nvs Bernoulli:')
    for bname, bacc in zip(['Prop-rand', 'Always-majority'], [prop_accuracy, majority_class_accuracy]):
        p_value = stats.binomtest(int(accuracy * len(gts_test)), n=len(gts_test), p=bacc).pvalue
        print(f"\t{bname}: {bacc:.4f}, p-value: {p_value:.6f}")
    print('\nvs MC:')
    for bname, base_preds in zip(['GT-distribution', 'pred-distribution'], [gts_test, preds]):
        fake_preds = np.stack([np.random.permutation(base_preds) for _ in range(10000)])
        fake_accs = (fake_preds==gts_test).mean(axis=1)
        p_value = ((fake_preds==gts_test).mean(axis=1) >= accuracy).mean()
        print(f"\t{bname}: {fake_accs.mean():.4f}, p-value: {p_value:.5f}")

    f1 = f1_score(gts_test, preds, average='macro')
    return accuracy, f1

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dirs', type=str, nargs='+', default='/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/with_others')
    parser.add_argument('--data-dirs-test', nargs='+', type=str)
    parser.add_argument('--feats', type=str, nargs='+', default='loglinpow-24-100')
    parser.add_argument('--ags', type=str, nargs='+', default=['mean', 'std', 'cnvrg', 'first', 'last'])
    parser.add_argument('--exclude-classes', type=str, nargs='+', default=[])
    parser.add_argument('--ags-test', type=str)
    parser.add_argument('--class-column', type=str, default='Context')
    parser.add_argument('--split-by', type=str, default='conv', choices=['conv', 'call'])
    parser.add_argument('--ml-model', type=str, default='lr', choices=['lr', 'random-forest'])
    parser.add_argument('--target', type=str, default='context', choices=['context', 'rank', 'rel-rank', 'caller', 'family'])
    parser.add_argument('--others-fams', type=str, choices=['other_EB', 'non_EB', 'all'])
    parser.add_argument('--frame-mfcc-std', action='store_true')
    parser.add_argument('--balance-classes', action='store_true')
    parser.add_argument('--test-on-held-out-callers', action='store_true')
    parser.add_argument('--fit-on-testset', action='store_true')
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--exclude-novel-test', action='store_true')
    parser.add_argument('--full-cross-train', action='store_true')
    parser.add_argument('--kfold-within-train', action='store_true')
    parser.add_argument('--show-plots', action='store_true')
    parser.add_argument('--n-iter', type=int, default=10000)
    parser.add_argument('--n-feat-importances', type=int, default=3)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-rank-bins', type=int, default=3)
    parser.add_argument('--small-class-thresh', type=int, default=5)
    ARGS = parser.parse_args()

    if ARGS.data_dirs_test is None:
        ARGS.data_dirs_test = ARGS.data_dirs
    if ARGS.ags_test is None:
        ARGS.ags_test = ARGS.ags

    if 'non_EB' in ARGS.data_dirs_test:
        assert ARGS.others_fams is not None

    if ARGS.full_cross_train:
        setting_combos = [
            ('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/with_others', ['first']),
            ('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/with_others', ['last']),
            ('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/with_others', ['mean']),
            ('/Users/sara/Documents/Research/Elephants/Joyce_July2025/cleaned_shifted8_conv/alone', ['first']),
            ]
        accs_mat_interp = np.empty([len(setting_combos), len(setting_combos)])
        f1s_mat_interp = np.empty([len(setting_combos), len(setting_combos)])
        accs_mat_full = np.empty([len(setting_combos), len(setting_combos)])
        f1s_mat_full = np.empty([len(setting_combos), len(setting_combos)])
        reload = False
        if reload:
            accs_mat_interp = np.load('LR_results/cross_settings/accs_mat.npy')
            f1s_mat_interp = np.load('LR_results/cross_settings/f1s_mat.npy')
            accs_mat_full_feats = np.load('LR_results/cross_settings/accs_mat.npy')
            f1s_mat_full_feats = np.load('LR_results/cross_settings/f1s_mat.npy')
        else:
            for i, (train_data_dir, train_ags) in enumerate(setting_combos):
                for j, (data_dir_test, test_ags) in enumerate(setting_combos):
                    train_metadata_fp = train_data_dir + '_metadata.csv'
                    test_metadata_fp = data_dir_test + '_metadata.csv'
                    tr_interp = ConversationLearner(train_data_dir, train_metadata_fp, data_dir_test, test_metadata_fp, ['loglinpow-24-100'], train_ags, test_ags, ARGS.target, ml_model=ARGS.ml_model, n_iter=ARGS.n_iter, balance_classes=ARGS.balance_classes, split_by=ARGS.split_by, test_on_held_out_callers=ARGS.test_on_held_out_callers, n_rank_bins=ARGS.n_rank_bins, small_class_thresh=ARGS.small_class_thresh, kfold_within_train=False)
                    f1_interp, acc_interp = tr_interp.train()
                    accs_mat_interp[i, j] = acc_interp
                    f1s_mat_interp[i, j] = f1_interp

                    tr_full = ConversationLearner(train_data_dir, train_metadata_fp, data_dir_test, test_metadata_fp, ['acoustic-contour-feats', 'mfcc-13'], train_ags, test_ags, ARGS.target, ml_model=ARGS.ml_model, n_iter=ARGS.n_iter, balance_classes=ARGS.balance_classes, split_by=ARGS.split_by, test_on_held_out_callers=ARGS.test_on_held_out_callers, n_rank_bins=ARGS.n_rank_bins, small_class_thresh=ARGS.small_class_thresh, kfold_within_train=False)
                    f1_full, acc_full = tr_full.train()
                    if np.isnan(f1_full).any():
                        breakpoint()
                    print('full:', train_metadata_fp, test_metadata_fp, acc_full, f1_full)
                    accs_mat_full[i, j] = acc_full
                    f1s_mat_full[i, j] = f1_full
            np.save('LR_results/cross_settings/accs_mat_interp.npy', accs_mat_interp)
            np.save('LR_results/cross_settings/f1s_mat_interp.npy', f1s_mat_interp)
            np.save('LR_results/cross_settings/accs_mat_full.npy', accs_mat_full)
            np.save('LR_results/cross_settings/f1s_mat_full.npy', f1s_mat_full)

        x = np.arange(len(setting_combos))
        print('len settings_combos = ', len(setting_combos))
        width = 0.35

        fig, axes = plt.subplots(2,2, figsize=(10,6))
        setting_names = [f'{os.path.basename(dirname)}-{ags[0]}'.replace('with_others', 'wothers') for dirname, ags in setting_combos]
        for scores, ax, metric_name in zip([accs_mat_interp, accs_mat_full, f1s_mat_interp, f1s_mat_full], axes.flatten(), ['Accuracy | Interpretable Features', 'Accuracy | Full Features', 'Macro-F1 | Interpretable Features', 'Macro-F1 | Full Features']):
            print(scores.shape, ax, metric_name, 'nans', np.isnan(scores).any())
            for j in range(len(setting_combos)):
                ax.bar(x + j*width/len(setting_combos), scores[:, j], width/len(setting_combos), label=f'test {setting_names[j]}')

            ax.set_xticks(x + width / 2)
            ax.set_xticklabels([f'train {setting_names[i]}' for i in range(len(setting_combos))], rotation=45, fontsize=8)
            ax.set_title(metric_name)
        handles, labels = axes[0,0].get_legend_handles_labels()
        ymin = min(ax.get_ylim()[0] for ax in axes.flat)
        ymax = max(ax.get_ylim()[1] for ax in axes.flat)
        for ax in axes.flat:
            ax.set_ylim(ymin, ymax)

        fig.legend(handles, labels, loc='upper center', ncol=4, fontsize='small')
        for ax in axes.flatten():
            ax.tick_params(labelsize=8)
            ax.set_title(ax.get_title(), fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(out_fp:='LR_results/cross_settings/cross_setting_f1s_accs.png')
        if ARGS.show_plots:
            os.system(f'/usr/bin/xdg-open {out_fp}')

    else:
        metadata_fps = [x + '_metadata.csv' for x in ARGS.data_dirs]
        test_metadata_fps = [x + '_metadata.csv' for x in ARGS.data_dirs_test]
        tr = ConversationLearner(ARGS.data_dirs, metadata_fps, ARGS.data_dirs_test, test_metadata_fps, ARGS.feats, ARGS.ags, ARGS.ags_test, ARGS.target, ml_model=ARGS.ml_model, n_iter=ARGS.n_iter, balance_classes=ARGS.balance_classes, split_by=ARGS.split_by, test_on_held_out_callers=ARGS.test_on_held_out_callers, n_rank_bins=ARGS.n_rank_bins, small_class_thresh=ARGS.small_class_thresh, kfold_within_train=ARGS.kfold_within_train)
        tr.train()
