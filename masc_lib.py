"""Shared decoding utilities.

The `segment`, `correlate`, `decod`, and `plot` functions below are adapted
from kingjr/meg-masc's `check_decoding.py` (BSD-3-Clause, Copyright (c)
2022 Jean-Remi King) -- see LICENSE. `build_metadata` is that script's
annotation-parsing logic pulled out on its own so it can be reused without
re-loading full sensor data (used by `surprisal.py`). `decod_per_fold` and
the surprisal wiring in `build_metadata` are new.
"""

from pathlib import Path

import mne
import mne_bids
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale
from tqdm import trange
from wordfreq import zipf_frequency

mne.set_log_level(False)

PH_INFO = pd.read_csv(Path(__file__).parent / "phoneme_info.csv")


def build_metadata(raw, surprisal_lookup=None, task=None):
    """Turn a BIDS raw's annotations into a word/phoneme-level dataframe.

    Each call covers a single task recording, so `word_order` is just
    that recording's word sequence (0, 1, 2, ...). If `surprisal_lookup`
    is given, it must be a dict mapping (task:int, word_order:int) ->
    surprisal (nats) -- `task` is then required -- and a "surprisal"
    column is attached the same way "wordfreq" is: on the phoneme
    immediately following each word.
    """
    meta = list()
    for annot in raw.annotations:
        d = eval(annot.pop("description"))
        for k, v in annot.items():
            assert k not in d.keys()
            d[k] = v
        meta.append(d)
    meta = pd.DataFrame(meta)
    meta["intercept"] = 1.0

    # compute voicing
    phonemes = meta.query('kind=="phoneme"')
    assert len(phonemes)
    for ph, d in phonemes.groupby("phoneme"):
        ph = ph.split("_")[0]
        match = PH_INFO.query("phoneme==@ph")
        assert len(match) == 1
        meta.loc[d.index, "voiced"] = match.iloc[0].phonation == "v"

    # compute word frequency and merge w/ phoneme
    meta["is_word"] = False
    words = meta.query('kind=="word"').copy()
    assert len(words) > 10
    meta.loc[words.index + 1, "is_word"] = True
    wfreq = lambda x: zipf_frequency(x, "en")  # noqa
    meta.loc[words.index + 1, "wordfreq"] = words.word.apply(wfreq).values

    if surprisal_lookup is not None:
        assert task is not None, "task is required when surprisal_lookup is given"
        words = words.reset_index().rename(columns={"index": "orig_index"})
        words["word_order"] = range(len(words))
        surp = words["word_order"].apply(
            lambda word_order: surprisal_lookup.get((int(task), int(word_order)))
        )
        meta.loc[words.orig_index.values + 1, "surprisal"] = surp.values

    meta = meta.query('kind=="phoneme"')
    assert len(meta.wordfreq.unique()) > 2
    return meta


def segment(raw, surprisal_lookup=None, task=None):
    meta = build_metadata(raw, surprisal_lookup=surprisal_lookup, task=task)

    events = np.c_[
        meta.onset * raw.info["sfreq"], np.ones((len(meta), 2))
    ].astype(int)

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-0.200,
        tmax=0.6,
        decim=10,
        baseline=(-0.2, 0.0),
        metadata=meta,
        preload=True,
        event_repeated="drop",
    )

    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    th = np.percentile(np.abs(epochs._data), 95)
    epochs._data[:] = np.clip(epochs._data, -th, th)
    epochs.apply_baseline()
    return epochs


def correlate(X, Y):
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)


def _binarize(y):
    """Median-split into two classes, unless y is already binary (e.g.
    "voiced"), in which case it's just cleaned up into plain 0/1 ints --
    a couple of raw float class labels (as z-scoring would otherwise
    leave them) trip up sklearn's label-type checks in some code paths.
    """
    y = np.asarray(y, dtype=float)
    uniques = np.unique(y)
    if len(uniques) <= 2:
        return (y == uniques[-1]).astype(int)
    y = scale(y[:, None])[:, 0]
    return (y > np.nanmedian(y)).astype(int)


def decod(X, y, meta, times):
    """Original meg-masc decoding: cross_val_predict + per-label correlation."""
    assert len(X) == len(y) == len(meta)
    meta = meta.reset_index()
    y = _binarize(y)

    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = KFold(5, shuffle=True, random_state=0)

    n, nchans, ntimes = X.shape
    preds = np.zeros((n, ntimes))
    for t in trange(ntimes):
        preds[:, t] = cross_val_predict(
            model, X[:, :, t], y, cv=cv, method="predict_proba"
        )[:, 1]

    out = list()
    for label, m in meta.groupby("label"):
        Rs = correlate(y[m.index, None], preds[m.index])
        for t, r in zip(times, Rs):
            out.append(dict(score=r, time=t, label=label, n=len(m.index)))
    return pd.DataFrame(out)


def decod_per_fold(X, y, times, n_splits=5, random_state=0):
    """Same decoding as `decod`, but returns one score curve per CV fold
    instead of a single pooled correlation, so the (fold x time) matrix
    can feed a cluster-based permutation test.
    """
    y = _binarize(y)
    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = KFold(n_splits, shuffle=True, random_state=random_state)

    n, nchans, ntimes = X.shape
    fold_scores = np.zeros((n_splits, ntimes))
    for fold_i, (train, test) in enumerate(cv.split(X)):
        for t in range(ntimes):
            model.fit(X[train, :, t], y[train])
            proba = model.predict_proba(X[test, :, t])[:, 1]
            fold_scores[fold_i, t] = correlate(y[test, None], proba[:, None])[0]
    return fold_scores


def plot(result):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, figsize=[6, 6])
    sns.lineplot(x="time", y="score", data=result, hue="label", ax=ax)
    ax.axhline(0, color="k")
    return fig


class Paths:
    def __init__(self, bids_root):
        self.bids = Path(bids_root)
        assert self.bids.exists(), f"missing BIDS root: {self.bids}"


def get_epochs(subject, bids_root, surprisal_lookup=None, sessions=(0,), tasks=(0, 1, 2, 3)):
    """Defaults to session 0 only: on a laptop, loading a full ~30min raw
    MEG recording peaks around 3-4GB per file before it's freed, so this
    keeps peak memory to roughly one recording at a time. Pass
    sessions=(0, 1) for the full two-session replication.
    """
    paths = Paths(bids_root)
    all_epochs = list()
    for session in sessions:
        for task in tasks:
            bids_path = mne_bids.BIDSPath(
                subject=subject,
                session=str(session),
                task=str(task),
                datatype="meg",
                root=paths.bids,
            )
            try:
                raw = mne_bids.read_raw_bids(bids_path)
            except FileNotFoundError:
                continue
            raw = raw.pick_types(meg=True, misc=False, eeg=False, eog=False, ecg=False)
            raw.load_data().filter(0.5, 30.0, n_jobs=1)
            epochs = segment(raw, surprisal_lookup=surprisal_lookup, task=task)
            epochs.metadata["task"] = task
            epochs.metadata["session"] = session
            all_epochs.append(epochs)
            del raw
    if not all_epochs:
        return None
    epochs = mne.concatenate_epochs(all_epochs)
    m = epochs.metadata
    epochs.metadata["label"] = "t" + m.task.astype(str) + "_s" + m.session.astype(str)
    return epochs
