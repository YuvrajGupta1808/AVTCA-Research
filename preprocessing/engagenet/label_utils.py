#!/usr/bin/env python3
"""Helpers for reading EngageNet label manifests."""

from __future__ import annotations

import csv
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


ENGAGENET_LABEL_MAP = {
    "Not-Engaged": 0,
    "Barely-engaged": 1,
    "Engaged": 2,
    "Highly-Engaged": 3,
}

SKIP_LABELS = {
    "SNP(Subject Not Present)",
}

SPLIT_LABEL_FILES = {
    "training": "train_engagement_labels.xlsx",
    "validation": "validation_engagement_labels.xlsx",
    "testing": "test_engagement_labels.csv",
}

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _column_letters(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_root.findall("a:si", NS):
            text = "".join(node.text or "" for node in item.iterfind(".//a:t", NS))
            shared_strings.append(text)

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[str, str]] = []
        for row in sheet_root.findall(".//a:sheetData/a:row", NS):
            values: dict[str, str] = {}
            for cell in row.findall("a:c", NS):
                cell_ref = cell.get("r", "")
                column = _column_letters(cell_ref)
                raw_value = cell.find("a:v", NS)
                value = raw_value.text if raw_value is not None else ""
                if cell.get("t") == "s" and value:
                    value = shared_strings[int(value)]
                values[column] = value
            rows.append(values)

    if not rows:
        return []

    header_row = rows[0]
    chunk_col = next(col for col, value in header_row.items() if value.strip().lower() == "chunk")
    label_col = next(col for col, value in header_row.items() if value.strip().lower() == "label")
    return [
        {
            "chunk": row.get(chunk_col, "").strip(),
            "label": row.get(label_col, "").strip(),
        }
        for row in rows[1:]
        if row.get(chunk_col, "").strip()
    ]


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "chunk": row["chunk"].strip(),
                "label": row["label"].strip(),
            }
            for row in reader
            if row.get("chunk", "").strip()
        ]


def load_split_labels(data_root: Path, subset: str) -> dict[str, int]:
    filename = SPLIT_LABEL_FILES[subset]
    path = data_root / filename
    if path.suffix.lower() == ".csv":
        rows = _read_csv_rows(path)
    else:
        rows = _read_xlsx_rows(path)

    labels: dict[str, int] = {}
    for row in rows:
        label_name = row["label"]
        if label_name in SKIP_LABELS:
            continue
        if label_name not in ENGAGENET_LABEL_MAP:
            raise ValueError(f"Unsupported EngageNet label: {label_name}")
        labels[row["chunk"]] = ENGAGENET_LABEL_MAP[label_name]
    return labels


def load_all_labels(data_root: Path) -> dict[str, dict[str, int]]:
    return {
        subset: load_split_labels(data_root, subset)
        for subset in SPLIT_LABEL_FILES
    }
