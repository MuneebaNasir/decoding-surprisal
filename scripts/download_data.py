"""Download one subject's worth of the MEG-MASC dataset from OSF.

MEG-MASC (Gwilliams, King et al., 2022) is hosted on OSF project `ag3kj`.
This pulls a single subject (both sessions) plus the top-level BIDS
sidecar files needed by mne-bids, instead of the full ~150GB dataset.

Usage:
    python scripts/download_data.py --subject 04 --out data/bids_anonym
"""

import argparse
import time
from pathlib import Path

from osfclient.api import OSF

PROJECT_ID = "ag3kj"
ROOT_FILES = {"dataset_description.json", "participants.tsv", "README.txt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="04", help="subject id, e.g. 04")
    parser.add_argument("--out", default="data/bids_anonym")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    osf = OSF()
    project = osf.project(PROJECT_ID)
    storage = project.storage("osfstorage")

    sub_prefix = f"/sub-{args.subject}/"
    wanted = []
    for f in storage.files:
        if f.path.startswith(sub_prefix) or f.path.lstrip("/") in ROOT_FILES:
            wanted.append(f)

    if not wanted:
        raise SystemExit(
            f"No files found for subject {args.subject}. "
            "Check the subject id exists in this OSF component."
        )

    print(f"Found {len(wanted)} files to download for subject {args.subject}.")
    for f in wanted:
        local_path = out_dir / f.path.lstrip("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        if local_path.exists() and local_path.stat().st_size > 0:
            print(f"  skip (exists): {local_path}")
            continue
        print(f"  downloading: {f.path}  ({(f.size or 0) / 1e6:.1f} MB)")
        for attempt in range(4):
            try:
                with open(local_path, "wb") as fh:
                    f.write_to(fh)
                break
            except RuntimeError as e:
                local_path.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                wait = 2 ** attempt
                print(f"    retry {attempt + 1}/3 after error ({e}); waiting {wait}s")
                time.sleep(wait)
                # re-fetch a fresh signed download URL before retrying
                f = next(x for x in storage.files if x.path == f.path)

    print("Done.")


if __name__ == "__main__":
    main()
