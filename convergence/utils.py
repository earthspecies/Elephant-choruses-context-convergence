import os
import glob
import numpy as np
import pandas as pd
from natsort import natsorted
import re
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def save_and_plot_results(y_true, y_pred, label_encoder, expname, do_open=True, suffix=''):
    os.makedirs(f'LR_results/{expname}', exist_ok=True)
    classif_report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    print(classif_report)
    with open(f'LR_results/{expname}/{expname}_classif_report-{suffix.strip()}.txt', 'w') as f:
        f.write(classif_report)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    abbrv_classes = [x[:3]+'.' for x in label_encoder.classes_]
    plt.xticks(ticks=np.arange(len(label_encoder.classes_)), labels=abbrv_classes, rotation=45, fontsize=8)
    plt.subplots_adjust(left=0.35)
    plt.title(f"Confusion Matrix{suffix}")
    conf_mat_fp = f'LR_results/{expname}/{expname}_conf_mat{suffix.replace(" ","-")}.png'
    save_open(conf_mat_fp, do_open=do_open)

# seems that we never call this?
def load_conv_splits(fp_base, ARGS):
    if ARGS.louis_flag:
        paths = glob.glob(f'joyce_poole_elephants/cleaned_shifted8_comv/with_others/{fp_base}_[0-9]*.wav')
    else:
        paths = glob.glob(f'joyce_poole_elephants/cleaned_shifted_8/with_others/{fp_base}_[0-9]*.wav')
    paths = natsorted(paths)
    splits = [librosa.load(x, sr=None)[0] for x in paths]
    return splits

def get_conv_splits(audio_fp, enforce_no_overlaps):

    print('Processing', audio_fp)

    y, sr = librosa.load(audio_fp, sr=None)
    fp_base = audio_fp.removesuffix('.wav')
    dirname = os.path.dirname(audio_fp)
    split_f0_fps = natsorted([os.path.join(dirname, x) for x in os.listdir(dirname) if re.search(r'_\d+\.csv$', x)])
    split_f0_fps = [x for x in split_f0_fps if x.startswith(fp_base)]
    
    splits = []
    split_f0s = [pd.read_csv(x) for x in split_f0_fps]

    print(f'Found {len(split_f0s)} split f0 csvs for {audio_fp}')
    # get all start and end times
    unordered_starts_ends = []
    for i, sdf in enumerate(split_f0s):
        split_start_temp = sdf.time.iloc[0]
        split_end_temp = sdf.time.iloc[-1]
        unordered_starts_ends.append((split_start_temp, split_end_temp))

    #print('unordered starts and ends:',unordered_starts_ends)
    # reorder by start time using indices
    sorted_indices = sorted(range(len(unordered_starts_ends)), key=lambda i: unordered_starts_ends[i][0])
    split_f0s = [split_f0s[i] for i in sorted_indices]
    split_f0_fps = [split_f0_fps[i] for i in sorted_indices]#print('Reordered starts and ends:', [(x.time.iloc[0], x.time.iloc[-1]) for x in split_f0s])
    #print('reordered split_f0s:', split_f0_fps)

    if any(x.empty for x in split_f0s):
        breakpoint()
    if len([x for x in split_f0s if not x.empty]) <= 1:
        return 'only one csv non-empty', []
    prev_split_end = 0
    prev_split_start = -1
    files_returned = []
    for i, sdf in enumerate(split_f0s):
        print(i)

        if sdf.empty:
            print(" Empty sdf found, skipping")
            continue

        # Determine split boundaries
        if i == len(split_f0s) - 1:  # Last segment
            split_start = sdf.time.iloc[0]
            split_end = sdf.time.iloc[-1]
        else:
            sdf_next = split_f0s[i+1]
            if sdf_next.empty:
                split_start = sdf.time.iloc[0]
                split_end = sdf.time.iloc[-1]
            elif enforce_no_overlaps:
                split_start = max(prev_split_end, sdf.time.iloc[0])
                split_end = min(sdf.time.iloc[-1], sdf_next.time.iloc[0])
                if split_end <= split_start:
                    print(f'complete overlap, skip this file')
                    continue
            else:
                split_start = sdf.time.iloc[0]
                split_end = sdf.time.iloc[-1]
                

        y_split = y[int(split_start*sr):int(split_end*sr)]
        if len(y_split) == 0:
            print(f' Split {i+1} has zero length, skipping')
            continue
        prev_split_end = sdf.time.iloc[-1]
        prev_split_start = split_start
        splits.append(y_split)
        files_returned.append(split_f0_fps[i])
        print(f' Call {i+1}: start {split_start:.2f}s, end {split_end:.2f}s, length {len(y_split)/sr:.2f}s')
        
    print(f' Extracted {len(splits)} splits from {audio_fp}')
    print('Final order check:')
    for i, fp in enumerate(files_returned):
        print(f'  Split {i+1}: {os.path.basename(fp)}')
    
    return splits, files_returned


def save_open(fp, do_open):
    plt.savefig(fp)
    if do_open:
        os.system(f'/usr/bin/xdg-open {fp}')

def filter_metadata_csv(metadata_fp, remove_dict=None):
    metadata = pd.read_csv(metadata_fp, index_col=0)
    metadata = metadata.loc[~metadata.SndFile.isna()]
    metadata = metadata.loc[~metadata.ContextType.isna()]
    metadata = metadata.loc[metadata.ContextType!='unknown']

    if remove_dict is not None:
        for k,v in remove_dict.items():
            for to_not_contain in v:
                metadata = metadata.loc[~metadata[k].str.contains(to_not_contain, na=False)]
    return metadata
