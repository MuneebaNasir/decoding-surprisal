"""Group-level version of extended_decoding.py: one decoding curve per
subject (not per CV fold), then a cluster-based permutation test across
subjects. This is the statistically correct version of the test -- CV
folds from one subject aren't independent observations, but subjects
are, so a real effect has a real chance of surviving correction here in
a way it structurally couldn't with a single subject.

Usage:
    python group_decoding.py --subjects 03,04,05,06,08 \
        --surprisal results/surprisal_sub-04.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.stats import permutation_cluster_1samp_test

from masc_lib import decod_subject, get_epochs
from surprisal import load_surprisal_lookup


def subject_curves(subject, bids_root, lookup, sessions, cache_dir):
    """Cached per-subject AND per-contrast: whatever kills these runs lands
    mid-decoding, not mid-loading, so caching each of the three contrasts
    separately (not just the finished subject) means a retry only ever
    redoes the one contrast that was interrupted, not the ~1-2min raw
    load + epoching each time too.
    """
    cache_file = cache_dir / f"sub-{subject}.npz"
    if cache_file.exists():
        data = np.load(cache_file)
        times = data["times"]
        out = {k: data[k] for k in ("wordfreq", "voiced", "surprisal")}
        print(f"  (from cache)", flush=True)
        return times, out

    partial_dir = cache_dir / f"_partial_sub-{subject}"
    contrast_files = {c: partial_dir / f"{c}.npy" for c in ("wordfreq", "voiced", "surprisal")}
    if all(f.exists() for f in contrast_files.values()) and (partial_dir / "times.npy").exists():
        times = np.load(partial_dir / "times.npy")
        out = {c: np.load(f) for c, f in contrast_files.items()}
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, times=times, **out)
        print(f"  (assembled from partial cache)", flush=True)
        return times, out

    epochs = get_epochs(subject, bids_root, surprisal_lookup=lookup, sessions=sessions)
    if epochs is None:
        return None
    words = epochs["is_word"]
    phonemes = epochs["not is_word"]
    times = epochs.times
    mask = ~pd.isna(words.metadata["surprisal"].values)

    contrast_inputs = {
        "wordfreq": (words.get_data() * 1e13, words.metadata["wordfreq"].values),
        "voiced": (phonemes.get_data() * 1e13, phonemes.metadata["voiced"].values),
        "surprisal": ((words.get_data() * 1e13)[mask], words.metadata["surprisal"].values[mask]),
    }

    partial_dir.mkdir(parents=True, exist_ok=True)
    np.save(partial_dir / "times.npy", times)
    out = {}
    for name, (X, y) in contrast_inputs.items():
        part_file = contrast_files[name]
        if part_file.exists():
            out[name] = np.load(part_file)
            print(f"    {name}: from partial cache", flush=True)
            continue
        out[name] = decod_subject(X, y, times)
        np.save(part_file, out[name])
        print(f"    {name}: done", flush=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, times=times, **out)
    return times, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="03,04,05,06,08")
    parser.add_argument("--bids-root", default="data/bids_anonym")
    parser.add_argument("--sessions", default="0", help="comma-separated, e.g. 0,1")
    parser.add_argument("--surprisal", default="results/surprisal_sub-04.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    subjects = args.subjects.split(",")
    sessions = tuple(int(s) for s in args.sessions.split(","))
    lookup = load_surprisal_lookup(args.surprisal)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "_cache"

    per_contrast = {"wordfreq": [], "voiced": [], "surprisal": []}
    times = None
    used_subjects = []
    for subject in subjects:
        print(f"=== subject {subject} ===", flush=True)
        result = subject_curves(subject, args.bids_root, lookup, sessions, cache_dir)
        if result is None:
            print(f"  no data for subject {subject}, skipping", flush=True)
            continue
        print(f"  done: {subject}", flush=True)
        times, curves = result
        for contrast, curve in curves.items():
            per_contrast[contrast].append(curve)
        used_subjects.append(subject)

    n_subjects = len(used_subjects)
    print(f"\ngroup analysis over {n_subjects} subjects: {used_subjects}")

    summary_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"wordfreq": "#7C8B86", "voiced": "#4A6FA5", "surprisal": "#C6791E"}

    for contrast, curves in per_contrast.items():
        mat = np.array(curves)  # (n_subjects, n_times)
        t_obs, clusters, cluster_p, _ = permutation_cluster_1samp_test(
            mat, n_permutations=2**n_subjects, tail=0, seed=0, out_type="mask"
        )
        sig_mask = np.zeros(len(times), dtype=bool)
        for cl, p in zip(clusters, cluster_p):
            if p < 0.05:
                sig_mask[cl] = True

        summary_rows.append(dict(
            label=contrast,
            n_subjects=n_subjects,
            min_cluster_p=cluster_p.min() if len(cluster_p) else np.nan,
            n_significant_timepoints=int(sig_mask.sum()),
        ))

        color = colors.get(contrast)
        mean_curve = mat.mean(0)
        sem = mat.std(0, ddof=1) / np.sqrt(n_subjects)
        ax.plot(times, mean_curve, label=contrast, color=color)
        ax.fill_between(times, mean_curve - sem, mean_curve + sem, color=color, alpha=0.15)
        if sig_mask.any():
            ax.scatter(
                times[sig_mask],
                np.full(sig_mask.sum(), -0.02 - 0.01 * list(colors).index(contrast)),
                s=6, color=color,
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"group_summary_n{n_subjects}.csv", index=False)
    print(summary.to_string(index=False))

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("decoding score (correlation), mean ± SEM across subjects")
    ax.legend()
    ax.set_title(f"group (n={n_subjects}): dots = cluster p<0.05 across subjects")
    fig.tight_layout()
    fig.savefig(out_dir / f"group_decoding_n{n_subjects}.png", dpi=150)
    print(f"wrote results to {out_dir}")


if __name__ == "__main__":
    main()
