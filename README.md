# Decoding Surprisal

Extends [`kingjr/meg-masc`](https://github.com/kingjr/meg-masc) — the MEG-MASC
dataset and decoding code released by Gwilliams, King, et al. — with one new
predictor and the significance testing the released script doesn't do.

Their `check_decoding.py` asks whether MEG activity, time-locked to word and
phoneme onsets, encodes **word frequency** and **phoneme voicing**. This adds
a third question, motivated by their own later paper on predictive coding in
speech comprehension ([Caucheteux, Gramfort & King, 2023, *Nat. Hum.
Behav.*](https://www.nature.com/articles/s41562-022-01516-2)): does it also
encode **word surprisal** — how unpredictable a word is given what came
before, estimated with GPT-2?

## What's original code vs. new

| File | What it is |
|---|---|
| `masc_lib.py` | `segment`, `correlate`, `decod`, `plot` — adapted from their `check_decoding.py` (BSD-3-Clause). `build_metadata` factors out their annotation-parsing so it's reusable; `decod_per_fold` is new. |
| `baseline_decoding.py` | Their two original analyses (word frequency, phoneme voicing), unmodified in substance. |
| `surprisal.py` | New: computes GPT-2 word surprisal from the story transcripts already in the BIDS metadata. |
| `extended_decoding.py` | New: wires surprisal in as a third contrast, then runs a cluster-based permutation test on all three — their script plots raw scores with no significance test at all. |

## Data

One subject (`sub-04`), both sessions, from
[MEG-MASC](https://github.com/kingjr/meg-masc) (OSF project `ag3kj`), not
the full 27-subject/~150GB release — enough trials for a real decoding
curve on a laptop.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --subject 04 --out data/bids_anonym
```

## Run

```
python baseline_decoding.py --subject 04 --sessions 0,1
python surprisal.py --subject 04 --out results/surprisal_sub-04.csv
python extended_decoding.py --subject 04 --sessions 0,1 --surprisal results/surprisal_sub-04.csv
```

`--sessions 0` runs on session 0 alone (faster, lower memory — useful for
a first smoke test); `results/*_ses-0.*` are that single-session run,
kept alongside the two-session results below for comparison.

## Results (sub-04, both sessions)

![decoding curves](results/extended_sub-04_ses-01.png)

| contrast | trials | best cluster p |
|---|---|---|
| word frequency | 17122 | 0.06 |
| phoneme voicing | 44400 | 0.06 |
| word surprisal | 17106 | 0.06 |

Pooling both of sub-04's listening sessions (vs. just session 0, in
`results/extended_sub-04.png`) sharpens the picture: word frequency
decoding now rises clearly from ~180ms and stays elevated through
400-500ms — close to where the published multi-subject curves peak —
and phoneme voicing shows a similar, smaller rise in the same window.
Surprisal stays flat around zero in *both* the one-session and
two-session runs — a stable null, not just noise on one run.

None of the three contrasts survive cluster correction at p<0.05 in
either run. That's expected, not a bug: with only 5 CV folds as
observations, the permutation test can never report a p-value below
1/32 ≈ 0.03 regardless of how many trials feed each fold — one
subject was never going to reach significance this way. Getting a real
answer on whether surprisal is encoded here would need a group-level
test across subjects, which is exactly the multi-subject design their
own papers use.

## Caveat, stated plainly

The permutation test uses this one subject's 5 cross-validation folds as
its observations — a laptop-scale stand-in for what should really be a
group-level test across subjects. It's enough to flag which time windows
are worth trusting, not a publication-grade significance claim. This is a
demonstration that the method runs correctly end to end, not a claim of a
novel neuroscience finding.

## Attribution

`phoneme_info.csv` and the core of `masc_lib.py` are from
[`kingjr/meg-masc`](https://github.com/kingjr/meg-masc), Copyright (c) 2022
Jean-Rémi King, BSD-3-Clause (see `LICENSE`). Data: Gwilliams et al., *MEG-MASC*,
[arXiv:2208.11488](https://arxiv.org/abs/2208.11488).
