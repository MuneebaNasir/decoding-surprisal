"""Extend meg-masc's decoding with a surprisal contrast, and add the
significance testing the released script doesn't do.

For each contrast (word frequency, phoneme voicing, word surprisal) this
computes a per-CV-fold decoding score curve and runs a cluster-based
permutation test (mne.stats.permutation_cluster_1samp_test) across time,
using the 5 folds as observations.

Caveat (stated here, not hidden): 5 folds is a small, single-subject
stand-in for what should really be a group-level test across subjects.
It's enough to flag which time windows are worth trusting, not a
publication-grade significance claim.

Usage:
    python extended_decoding.py --subject 04 --bids-root data/bids_anonym \
        --surprisal results/surprisal_sub-04.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test

from masc_lib import decod_per_fold, get_epochs
from surprisal import load_surprisal_lookup


def run_contrast(X, y, times, label):
    mask = ~pd.isna(y)
    X, y = X[mask], y[mask]
    fold_scores = decod_per_fold(X, y, times)
    t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
        fold_scores, n_permutations=1000, tail=0, seed=0, out_type="mask"
    )
    sig_mask = np.zeros(len(times), dtype=bool)
    for cl, p in zip(clusters, cluster_p):
        if p < 0.05:
            sig_mask[cl] = True
    return dict(
        label=label,
        times=times,
        mean_score=fold_scores.mean(0),
        sig_mask=sig_mask,
        cluster_p=cluster_p.min() if len(cluster_p) else np.nan,
        n_trials=mask.sum(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="04")
    parser.add_argument("--bids-root", default="data/bids_anonym")
    parser.add_argument("--sessions", default="0", help="comma-separated, e.g. 0,1")
    parser.add_argument("--surprisal", default="results/surprisal_sub-04.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    sessions = tuple(int(s) for s in args.sessions.split(","))
    tag = "".join(str(s) for s in sessions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lookup = load_surprisal_lookup(args.surprisal)
    epochs = get_epochs(args.subject, args.bids_root, surprisal_lookup=lookup, sessions=sessions)
    if epochs is None:
        raise SystemExit(f"no data found for subject {args.subject}")

    words = epochs["is_word"]
    phonemes = epochs["not is_word"]
    times = epochs.times

    contrasts = [
        run_contrast(words.get_data() * 1e13, words.metadata["wordfreq"].values, times, "wordfreq"),
        run_contrast(phonemes.get_data() * 1e13, phonemes.metadata["voiced"].values, times, "voiced"),
        run_contrast(words.get_data() * 1e13, words.metadata["surprisal"].values, times, "surprisal"),
    ]

    summary = pd.DataFrame(
        [{"label": c["label"], "n_trials": c["n_trials"],
          "min_cluster_p": c["cluster_p"],
          "n_significant_timepoints": int(c["sig_mask"].sum())}
         for c in contrasts]
    )
    summary.to_csv(out_dir / f"extended_summary_sub-{args.subject}_ses-{tag}.csv", index=False)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"wordfreq": "#7C8B86", "voiced": "#4A6FA5", "surprisal": "#C6791E"}
    for c in contrasts:
        color = colors.get(c["label"], None)
        ax.plot(c["times"], c["mean_score"], label=c["label"], color=color)
        if c["sig_mask"].any():
            ax.scatter(
                c["times"][c["sig_mask"]],
                np.full(c["sig_mask"].sum(), -0.05 - 0.02 * list(colors).index(c["label"])),
                s=4, color=color,
            )
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("decoding score (correlation)")
    ax.legend()
    ax.set_title(f"sub-{args.subject} ses-{tag}: baseline contrasts + surprisal, dots = cluster p<0.05")
    fig.tight_layout()
    fig.savefig(out_dir / f"extended_sub-{args.subject}_ses-{tag}.png", dpi=150)
    print(f"wrote results to {out_dir}")


if __name__ == "__main__":
    main()
