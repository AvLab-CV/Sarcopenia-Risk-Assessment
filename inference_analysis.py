import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def _parse_predictions_cell(x: object) -> np.ndarray:
    """
    The sliding-window CSV stores an array of per-window unstable probabilities in the
    `predictions` column. Depending on pandas/numpy versions it may look like:
      - "[0.1 0.2 0.3]"
      - "[0.1, 0.2, 0.3]"
      - "0.1 0.2 0.3"
    """
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    s = s.replace(",", " ")
    arr = np.fromstring(s, sep=" ")
    return arr


def _resolve_sliding_window_csv(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.is_dir():
        raise SystemExit(f"Path does not exist: {path}")

    matches = sorted(path.glob("sliding_window_windowsize*_stride*.csv"))
    if len(matches) == 0:
        raise SystemExit(
            f"No sliding-window CSV found in {path}. Expected a file named like "
            f"`sliding_window_windowsize64_stride8.csv`."
        )
    if len(matches) > 1:
        names = "\n  - " + "\n  - ".join(str(p) for p in matches)
        raise SystemExit(f"Multiple sliding-window CSVs found in {path}:{names}\nPlease pass one CSV explicitly.")
    return matches[0]


def _maybe_add_subject_column(df: pd.DataFrame, mapping_csv: Optional[Path]) -> pd.DataFrame:
    # Preferred: subject already in the CSV
    if "subject" in df.columns:
        return df.assign(subject=df["subject"].astype(int))

    # Next: mapping file for the public sarcopenia dataset (index -> subject id)
    if mapping_csv is not None:
        mapping_df = pd.read_csv(mapping_csv)
        if not {"index", "subject"}.issubset(mapping_df.columns):
            raise SystemExit("Mapping CSV must have 'index' and 'subject' columns.")
        index_to_subject = dict(zip(mapping_df["index"].astype(int), mapping_df["subject"].astype(int)))
        if "index" not in df.columns:
            raise SystemExit("Input CSV is missing 'index' column; can't apply mapping CSV.")
        subjects = df["index"].astype(int).map(index_to_subject)
        if subjects.isna().any():
            missing = df.loc[subjects.isna(), "index"].astype(int).unique().tolist()
            raise SystemExit(f"Missing subject mapping for index values: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        return df.assign(subject=subjects.astype(int))

    # Fallback: treat the row index as the subject identifier
    if "index" in df.columns:
        return df.assign(subject=df["index"].astype(int))
    return df.assign(subject=pd.RangeIndex(len(df)).astype(int))


def compute_subject_instability(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    if "predictions" not in df.columns:
        raise SystemExit("Input CSV must contain a 'predictions' column.")

    preds = df["predictions"].map(_parse_predictions_cell)
    unstable_count = preds.map(lambda a: int((a > threshold).sum()))
    stable_count = preds.map(lambda a: int((a <= threshold).sum()))

    denom = (stable_count + unstable_count).to_numpy()
    instability_rate = np.where(denom > 0, unstable_count / (stable_count + unstable_count), np.nan)

    out = df.copy()
    out["instability_rate"] = instability_rate
    out["risk_group"] = np.where(out["instability_rate"].to_numpy() >= 0.5, "mostly unstable", "mostly stable")
    out.loc[out["instability_rate"].isna(), "risk_group"] = "unknown"

    # Ensure a clip identifier exists for printing
    if "clip" not in out.columns:
        if "index" in out.columns:
            out["clip"] = out["index"].astype(str)
        else:
            out["clip"] = out.index.astype(str)

    # Print one row per clip (i.e., per input sample)
    cols = ["clip"]
    if "subject" in out.columns:
        cols.append("subject")
    cols += ["instability_rate", "risk_group"]
    return out[cols]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-subject instability rate (R) and sarcopenia risk group from a single "
            "sliding-window CSV produced by `skateformer/inference.py`."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a sliding-window CSV, or a work_dir containing exactly one such CSV.",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping `index` -> `subject` (e.g. `csvs/sliding_window_subject_idx_to_subject_id.csv`). "
            "Used if the input CSV does not already contain a `subject` column."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output CSV path. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold applied to per-window probability to count a window as unstable. Default: 0.5.",
    )
    args = parser.parse_args()

    csv_path = _resolve_sliding_window_csv(args.input)
    df = pd.read_csv(csv_path)
    df = _maybe_add_subject_column(df, args.mapping_csv)
    out_df = compute_subject_instability(df, threshold=args.threshold)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out, index=False)
        print(f"Wrote: {args.out}")
    else:
        # Compact deterministic printing
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
            print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
