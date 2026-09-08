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
| `extended_decoding.py` | New: wires surprisal in as a third contrast, then runs a within-subject cluster-based permutation test (folds as observations) — a fast single-subject sanity check. |
| `group_decoding.py` | New: the statistically correct version — `decod_subject` (also new, in `masc_lib.py`) gives one decoding curve per subject, then the cluster test runs across subjects, not folds. |

## Data

All 8 subjects (`03, 04, 05, 06, 08, 09, 10, 11`) available in this OSF
component of [MEG-MASC](https://github.com/kingjr/meg-masc) (project
`ag3kj`) — the full public release has 27 subjects across two OSF
components (~150GB); this is the first, laptop-sized (~29GB) one.

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py --subject 04 --out data/bids_anonym  # repeat per subject
```

## Run

```
python baseline_decoding.py --subject 04 --sessions 0,1
python surprisal.py --subject 04 --out results/surprisal_sub-04.csv
python extended_decoding.py --subject 04 --sessions 0,1 --surprisal results/surprisal_sub-04.csv
python group_decoding.py --subjects 03,04,05,06,08,09,10,11 --surprisal results/surprisal_sub-04.csv
```

`baseline_decoding.py`/`extended_decoding.py` are single-subject (use
`--sessions 0` for a faster, lower-memory smoke test; `results/*_ses-0.*`
vs `*_ses-01.*` are that comparison). `group_decoding.py` is the main
result — the group-level test across subjects, below.

## Results (group, n=8 subjects — all available in this OSF release)

![group decoding curves](results/group_decoding_n8.png)

| contrast | subjects | best cluster p | significant timepoints |
|---|---|---|---|
| word frequency | 8 | **0.008** | 46 / 81 |
| phoneme voicing | 8 | **0.008** | 28 / 81 |
| word surprisal | 8 | 0.15 | 0 / 81 |

The statistically correct version of the test: one decoding curve per
*subject* (not per CV fold), then a cluster-based permutation test across
subjects — real independent observations. At n=8, word frequency and
phoneme voicing both clear p<0.05 (marked as dots on the plot) — a real,
significant, group-level effect, not just a visible trend. Word frequency
is significant from ~100-550ms, voicing from ~90-390ms — both in the
expected post-word-onset window. Surprisal stays flat, mostly at or below
zero, across every sample size tested (5 and 8 subjects, 1 and 2
sessions) — a consistent, honest null, not an artifact of too little data.

With n=5 the same two effects were visible but fell just short of
significance (p=0.06 both) — see `results/group_decoding_n5.png` for
that intermediate step.

## Results (single subject: sub-04, both sessions)

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

8 subjects is still a small group by the standards of the field (the
original meg-masc paper used 27) — real, but on the smaller end. The
word-frequency and voicing effects are genuinely significant at this
sample size; the surprisal null should be read as "no effect detected
in 8 subjects," not "proven absent."

## Attribution

`phoneme_info.csv` and the core of `masc_lib.py` are from
[`kingjr/meg-masc`](https://github.com/kingjr/meg-masc), Copyright (c) 2022
Jean-Rémi King, BSD-3-Clause (see `LICENSE`). Data: Gwilliams et al., *MEG-MASC*,
[arXiv:2208.11488](https://arxiv.org/abs/2208.11488).
