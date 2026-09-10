"""Compute GPT-2 word surprisal for each MEG-MASC story.

Reads only the BIDS *annotations* (word text + onset order) for one
subject/session -- not the MEG sensor data -- reconstructs each of the
four stories in listening order, and scores every word's surprisal
(-log P(word | preceding context)) with a small pretrained GPT-2.
Multi-token words get the summed surprisal of their tokens.

Story text is identical across a subject's two listening sessions, so
this only needs to run once per subject; the result is keyed by
(task, word_order) and merged into the MEG metadata in extended_decoding.py.

Usage:
    python surprisal.py --subject 04 --bids-root data/bids_anonym \
        --out results/surprisal_sub-04.csv
"""

import argparse
from pathlib import Path

import mne_bids
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def token_surprisals(input_ids, model, window=896, stride=448):
    """Per-token surprisal (nats) via a sliding window, so texts longer
    than GPT-2's 1024-token context still get real preceding context.
    """
    n = len(input_ids)
    surprisal = np.full(n, np.nan)
    pos = 0
    while pos < n:
        end = min(pos + window, n)
        chunk = torch.tensor([input_ids[pos:end]])
        with torch.no_grad():
            logits = model(chunk).logits[0]
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        fill_from = 1 if pos == 0 else (window - stride)
        fill_from = min(fill_from, end - pos - 1) if end - pos > 1 else 0
        for i in range(fill_from, end - pos):
            tok_id = input_ids[pos + i]
            surprisal[pos + i] = -logprobs[i - 1, tok_id].item()
        if end == n:
            break
        pos += stride
    return surprisal


def story_words(subject, bids_root, session, task):
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session=str(session),
        task=str(task),
        datatype="meg",
        root=bids_root,
    )
    raw = mne_bids.read_raw_bids(bids_path, verbose=False)
    rows = []
    for annot in raw.annotations:
        annot = dict(annot)
        d = eval(annot.pop("description"))
        d.update(annot)
        rows.append(d)
    words = pd.DataFrame(rows)
    words = words.query('kind=="word"').sort_values("onset").reset_index(drop=True)
    return words


def compute_task_surprisal(words, tokenizer, model):
    word_list = words.word.astype(str).tolist()
    text = " ".join(word_list)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # char start of each word in the joined text
    word_starts = []
    pos = 0
    for w in word_list:
        word_starts.append(pos)
        pos += len(w) + 1  # +1 for the joining space

    tok_surprisal = token_surprisals(input_ids, model)

    word_surprisal = np.zeros(len(word_list))
    word_has_value = np.zeros(len(word_list), dtype=bool)
    starts_arr = np.array(word_starts)
    for tok_idx, (start, end) in enumerate(offsets):
        if end <= start:
            continue
        # GPT-2's tokenizer attaches the leading space to each word's
        # token (e.g. "stood" -> " stood", offset starting at the space
        # *before* it) -- using `start` here would land in the *previous*
        # word's span. `end - 1` always falls inside the right word.
        word_idx = int(np.searchsorted(starts_arr, end - 1, side="right") - 1)
        if np.isnan(tok_surprisal[tok_idx]):
            continue
        word_surprisal[word_idx] += tok_surprisal[tok_idx]
        word_has_value[word_idx] = True

    word_surprisal[~word_has_value] = np.nan
    word_surprisal[0] = np.nan  # first word of the story has no context
    return word_surprisal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="04")
    parser.add_argument("--bids-root", default="data/bids_anonym")
    parser.add_argument("--session", default="0")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--out", default="results/surprisal.csv")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.eval()

    rows = []
    for task in range(4):
        try:
            words = story_words(args.subject, args.bids_root, args.session, task)
        except FileNotFoundError:
            print(f"task-{task}: not found, skipping")
            continue
        surprisal = compute_task_surprisal(words, tokenizer, model)
        for order, (w, s) in enumerate(zip(words.word.tolist(), surprisal)):
            rows.append(dict(task=task, word_order=order, word=w, surprisal=s))
        print(f"task-{task}: {len(words)} words, "
              f"mean surprisal={np.nanmean(surprisal):.2f} nats")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


def load_surprisal_lookup(csv_path):
    df = pd.read_csv(csv_path)
    return {
        (int(r.task), int(r.word_order)): r.surprisal
        for r in df.itertuples()
        if not np.isnan(r.surprisal)
    }


if __name__ == "__main__":
    main()
