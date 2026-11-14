#!/usr/bin/env python3
"""Audit an annotation file's text column for label leakage.

Reports, per annotation file:
  * coverage (non-empty text rows) and unique-string count
  * caption->label determinism: fraction of text rows whose string maps to a
    single label (v1 chat text is ~99.9% deterministic = a label oracle)
  * mean per-string label entropy (weighted by string frequency)
  * bag-of-words logistic regression trained on the training split's text,
    evaluated on validation and testing rows with non-empty text, vs the
    majority-class baseline.

A legitimate input-derived text column should show *moderate* label correlation
(behavior genuinely correlates with engagement); what must not appear is v1's
near-100% string->label determinism.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def load_rows(path: Path) -> list[tuple[str, int, str]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle, delimiter=";"):
            if len(row) >= 4:
                rows.append((row[3], int(row[2]), row[4] if len(row) > 4 else ""))
    return rows


def audit(path: Path) -> None:
    rows = load_rows(path)
    text_rows = [(split, label, text) for split, label, text in rows if text]
    print(f"\n=== {path.name} ===")
    print(f"rows: {len(rows)}, non-empty text: {len(text_rows)} "
          f"({len(text_rows) / len(rows):.2%})")
    if not text_rows:
        print("no text column — nothing to audit")
        return

    by_string: dict[str, Counter] = defaultdict(Counter)
    for _, label, text in text_rows:
        by_string[text][label] += 1
    print(f"unique strings: {len(by_string)}")

    deterministic = sum(c.total() for c in by_string.values() if len(c) == 1)
    print(f"string->label deterministic rows: {deterministic}/{len(text_rows)} "
          f"({deterministic / len(text_rows):.2%})")

    total = sum(c.total() for c in by_string.values())
    entropy = sum(
        (c.total() / total)
        * -sum((n / c.total()) * math.log2(n / c.total()) for n in c.values())
        for c in by_string.values()
    )
    print(f"mean per-string label entropy: {entropy:.3f} bits "
          f"(0 = oracle; {math.log2(4):.1f} = uninformative over 4 labels)")

    train = [(t, l) for s, l, t in text_rows if s == "training"]
    heldout = {
        split: [(t, l) for s, l, t in text_rows if s == split]
        for split in ("validation", "testing")
    }
    if not train or not any(heldout.values()):
        print("missing train or held-out text rows — skipping classifier probe")
        return

    vectorizer = CountVectorizer(token_pattern=r"[a-z']+", lowercase=True)
    x_train = vectorizer.fit_transform([t for t, _ in train])
    y_train = np.array([l for _, l in train])
    clf = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    for split, pairs in heldout.items():
        if not pairs:
            continue
        y = np.array([l for _, l in pairs])
        acc = clf.score(vectorizer.transform([t for t, _ in pairs]), y)
        majority = np.bincount(y).max() / len(y)
        print(f"BoW logistic regression text->label acc [{split}]: {acc:.2%} "
              f"(majority baseline {majority:.2%})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", type=Path,
                        default=[Path("preprocessing/engagenet/annotations_engagement_a10.txt"),
                                 Path("preprocessing/engagenet/annotations_engagement_v2_a10.txt")])
    args = parser.parse_args()
    for path in args.files:
        audit(path)


if __name__ == "__main__":
    main()
