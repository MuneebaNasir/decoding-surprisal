"""Reproduce meg-masc's two published decoding analyses, unmodified:
word-frequency decoding and phoneme-voicing decoding, on one subject.

Usage:
    python baseline_decoding.py --subject 04 --bids-root data/bids_anonym
"""

import argparse
from pathlib import Path

import pandas as pd

from masc_lib import decod, get_epochs, plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="04")
    parser.add_argument("--bids-root", default="data/bids_anonym")
    parser.add_argument("--sessions", default="0", help="comma-separated, e.g. 0,1")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    sessions = tuple(int(s) for s in args.sessions.split(","))
    tag = "".join(str(s) for s in sessions)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = get_epochs(args.subject, args.bids_root, sessions=sessions)
    if epochs is None:
        raise SystemExit(f"no data found for subject {args.subject}")

    words = epochs["is_word"]
    X = words.get_data() * 1e13
    y = words.metadata["wordfreq"].values
    results_wf = decod(X, y, words.metadata, words.times)
    results_wf["contrast"] = "wordfreq"

    phonemes = epochs["not is_word"]
    X = phonemes.get_data() * 1e13
    y = phonemes.metadata["voiced"].values
    results_ph = decod(X, y, phonemes.metadata, phonemes.times)
    results_ph["contrast"] = "voiced"

    results = pd.concat([results_wf, results_ph], ignore_index=True)
    results["subject"] = args.subject
    results.to_csv(out_dir / f"baseline_sub-{args.subject}_ses-{tag}.csv", index=False)

    for contrast, sub in results.groupby("contrast"):
        fig = plot(sub)
        fig.savefig(out_dir / f"baseline_{contrast}_sub-{args.subject}_ses-{tag}.png", dpi=150)
    print(f"wrote results to {out_dir}")


if __name__ == "__main__":
    main()
