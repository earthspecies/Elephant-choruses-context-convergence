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
    os.makedirs(f'context_prediction_output/{expname}', exist_ok=True)
    classif_report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)
    # print(classif_report)
    with open(f'context_prediction_output/{expname}/{expname}_classif_report-{suffix.strip()}.txt', 'w') as f:
        f.write(classif_report)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    abbrv_classes = [x[:3]+'.' for x in label_encoder.classes_]
    plt.xticks(ticks=np.arange(len(label_encoder.classes_)), labels=abbrv_classes, rotation=45, fontsize=8)
    plt.subplots_adjust(left=0.35)
    plt.title(f"Confusion Matrix{suffix}")
    conf_mat_fp = f'context_prediction_output/{expname}/{expname}_conf_mat{suffix.replace(" ","-")}.png'
    save_open(conf_mat_fp, do_open=do_open)


def save_open(fp, do_open):
    plt.savefig(fp)
    if do_open:
        os.system(f'open {fp}')

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
