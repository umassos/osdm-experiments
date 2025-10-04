"""
trace_loader.py

Utility functions to load Alibaba trace data, bucketize into fixed windows,
compute per-bucket weighted active job counts, and generate randomized
base vs. flexible series for a contiguous time window of length T buckets.

Typical usage:

    from trace_loader import (
        prepare_exploded_from_csv,
        compute_base_flexible_series,
        contiguous_indexes,
    )

    # Build context once (15-minute buckets by default)
    ctx = prepare_exploded_from_csv("batch_task.csv", bucket_minutes=15)

    # Choose a starting bucket and length T (e.g., one day = 96 buckets)
    start_bucket = int(ctx["bucket_stats"]["bucket"].min())
    T = 96
    idxs = contiguous_indexes(start_bucket, T)

    base_scaled, flex_scaled, deltas, details = compute_base_flexible_series(
        ctx, day_bucket_indexes=idxs, T=T, seed=123, scale_divisor=40.0
    )

Design notes:
- We adhere to the bucket/overlap logic used in the notebook.
- "Weighted" counts are the sum of per-task fractional overlaps in a bucket,
  where a job's fractional overlap equals (overlap_seconds / bucket_seconds).
- The function `compute_base_flexible_series` performs a coin flip per unique
  job to assign it to base vs. flexible, then aggregates their contributions
  separately and returns two scaled series.
- T (length of window in buckets) is a parameter; no hard-coded 96 remains.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# ------------------------------
# Data loading and preparation
# ------------------------------

def load_trace_dataframe(csv_path: str) -> pd.DataFrame:
    """Load the raw trace CSV into a DataFrame.

    Expected columns include: start_time, end_time (Unix seconds).
    """
    df = pd.read_csv(csv_path)
    return df


def clean_trace_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning rules consistent with the notebook.

    - Drop rows with NaN start_time
    - Drop rows with end_time <= start_time
    - Drop rows with start_time == 0
    """
    out = df.copy()
    out = out[out["start_time"].notna()]
    out = out[out["end_time"] > out["start_time"]]
    out = out[out["start_time"] != 0]
    return out


def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add datetime conversions for start and end times from Unix seconds."""
    out = df.copy()
    out["datetime_start_time"] = pd.to_datetime(out["start_time"], unit="s")
    out["datetime_end_time"] = pd.to_datetime(out["end_time"], unit="s")
    return out


# ------------------------------
# Bucketing and exploding
# ------------------------------

def _row_bucket_overlap(
    s: pd.Timestamp,
    e: pd.Timestamp,
    trace_start: pd.Timestamp,
    bucket_seconds: int,
) -> list[tuple[int, float]]:
    """Compute list of (bucket_index, fractional_overlap) for a single job.

    Returns empty list if invalid or no overlap.
    """
    if pd.isna(s) or pd.isna(e) or e <= s:
        return []

    # Map to integer bucket indexes relative to trace_start
    start_idx = int(np.floor((s - trace_start).total_seconds() / bucket_seconds))
    end_idx = int(np.floor((e - trace_start).total_seconds() / bucket_seconds))

    overlaps: list[tuple[int, float]] = []
    for b in range(start_idx, end_idx + 1):
        b_start = trace_start + pd.Timedelta(seconds=b * bucket_seconds)
        b_end = b_start + pd.Timedelta(seconds=bucket_seconds)
        overlap_seconds = (min(e, b_end) - max(s, b_start)).total_seconds()
        if overlap_seconds > 0:
            frac = overlap_seconds / bucket_seconds
            overlaps.append((b, float(frac)))
    return overlaps


def build_exploded(
    df: pd.DataFrame,
    bucket_minutes: int = 15,
) -> tuple[pd.DataFrame, pd.Timestamp, int, pd.DataFrame]:
    """Build the exploded (task, bucket) view and bucket_stats.

    Returns: exploded, trace_start, bucket_seconds, bucket_stats
    - exploded columns: [bucket, task_index, overlap_frac]
    - bucket_stats columns: [bucket, n_active_jobs, weighted_active_jobs, bucket_start, bucket_end]
    """
    bucket_seconds = int(bucket_minutes * 60)

    # Align with the notebook logic: relative to the earliest start time
    trace_start = df["datetime_start_time"].min().floor("min")

    exploded_records: list[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        s = row["datetime_start_time"]
        e = row["datetime_end_time"]
        for b, frac in _row_bucket_overlap(s, e, trace_start, bucket_seconds):
            exploded_records.append(
                {
                    "bucket": b,
                    "task_index": idx,
                    "overlap_frac": frac,
                }
            )

    exploded = pd.DataFrame(exploded_records)
    if exploded.empty:
        raise ValueError("No overlapping records found; check timestamps and buckets.")

    bucket_stats = (
        exploded.groupby("bucket").agg(
            n_active_jobs=("task_index", "nunique"),
            weighted_active_jobs=("overlap_frac", "sum"),
        )
    ).reset_index()
    bucket_stats["bucket_start"] = bucket_stats["bucket"].apply(
        lambda b: trace_start + pd.Timedelta(seconds=b * bucket_seconds)
    )
    bucket_stats["bucket_end"] = bucket_stats["bucket_start"] + pd.Timedelta(seconds=bucket_seconds)
    bucket_stats = bucket_stats.sort_values("bucket").reset_index(drop=True)

    return exploded, trace_start, bucket_seconds, bucket_stats


def prepare_exploded_from_csv(
    csv_path: str,
    bucket_minutes: int = 15,
) -> Dict[str, Any]:
    """Full pipeline from CSV to an analysis context dictionary.

    Returns a dict with keys:
      - exploded: pd.DataFrame
      - trace_start: pd.Timestamp
      - bucket_seconds: int
      - bucket_minutes: int
      - bucket_stats: pd.DataFrame
      - df: cleaned DataFrame with datetime columns (for reference)
    """
    df = load_trace_dataframe(csv_path)
    df = clean_trace_dataframe(df)
    df = add_datetime_columns(df)
    exploded, trace_start, bucket_seconds, bucket_stats = build_exploded(df, bucket_minutes)
    return {
        "exploded": exploded,
        "trace_start": trace_start,
        "bucket_seconds": bucket_seconds,
        "bucket_minutes": bucket_minutes,
        "bucket_stats": bucket_stats,
        "df": df,
    }


# ------------------------------
# Base vs. Flexible computation
# ------------------------------

def contiguous_indexes(start_bucket: int, T: int) -> list[int]:
    """Helper to build a contiguous list of T bucket indices starting at start_bucket."""
    return list(range(int(start_bucket), int(start_bucket) + int(T)))


def compute_base_flexible_series(
    context: Dict[str, Any],
    day_bucket_indexes: Iterable[int],
    T: int,
    seed: int | None = 42,
    scale_divisor: float = 40.0,
    flexible_deadline_hours_choices: tuple[int, int, int] = (6, 12, 24),
    proportion_base: float = 0.5,
) -> Tuple[list[float], list[float], list[int], pd.DataFrame]:
    """
    Given a contiguous list of T bucket indices, split jobs into base vs flexible via
    a coin flip per unique job and compute two scaled T-length series of weighted contributions.

    Inputs:
      - context: dictionary returned by prepare_exploded_from_csv
      - day_bucket_indexes: iterable of T contiguous integer bucket indices
      - T: desired window length in buckets (e.g., 96 for a day @ 15-minute bins)
      - seed: RNG seed for reproducibility (None for non-deterministic)
      - scale_divisor: scale the series by dividing by this value
      - flexible_deadline_hours_choices: candidate deadlines (hours) for flexible load; used to
        generate per-bucket deadlines (optional, for downstream scheduling/simulation)

    Returns:
      - base_series_scaled: list[float] of length T
      - flexible_series_scaled: list[float] of length T
      - deltas: list[int] of length T; for each local bucket i in the window, the latest bucket index
        within [0..T-1] by which flexible load should be scheduled (based on a random deadline).
      - details_df: DataFrame with per-(bucket, job) contributions and assignment for this window
    """
    # Extract from context
    exploded = context.get("exploded")
    bucket_seconds = int(context.get("bucket_seconds"))
    bucket_minutes = int(context.get("bucket_minutes"))

    if not isinstance(exploded, pd.DataFrame):
        raise RuntimeError("Invalid context: missing 'exploded' DataFrame.")

    idxs = list(day_bucket_indexes)
    if len(idxs) != int(T):
        raise ValueError(f"Expected T={T} bucket indexes, got {len(idxs)}")

    # Ensure contiguity
    start = min(idxs)
    expected = list(range(start, start + int(T)))
    if sorted(idxs) != expected:
        raise ValueError("day_bucket_indexes must be T contiguous integers")

    day_exploded = exploded[exploded["bucket"].isin(idxs)].copy()
    if day_exploded.empty:
        raise ValueError("No overlapping records found for the provided bucket indexes")

    # Unique jobs and RNG
    unique_jobs = day_exploded["task_index"].unique()
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # Assign jobs: base vs flexible
    is_base = rng.random(len(unique_jobs)) < proportion_base
    job_to_group = {job: ("base" if b else "flexible") for job, b in zip(unique_jobs, is_base)}
    day_exploded["group"] = day_exploded["task_index"].map(job_to_group)

    # Aggregate overlap per (bucket, group)
    grouped = (
        day_exploded.groupby(["bucket", "group"])['overlap_frac']
        .sum()
        .unstack(fill_value=0.0)
    )

    for col in ["base", "flexible"]:
        if col not in grouped.columns:
            grouped[col] = 0.0

    grouped = grouped.reindex(index=expected, fill_value=0.0)

    base_series = grouped["base"].astype(float).tolist()
    flexible_series = grouped["flexible"].astype(float).tolist()

    # Scale outputs
    if scale_divisor == 0:
        base_scaled = base_series
        flexible_scaled = flexible_series
    else:
        base_scaled = [x / scale_divisor for x in base_series]
        flexible_scaled = [x / scale_divisor for x in flexible_series]


    # I want roughly 2 out of the T demands to have base > 0 and flexible > 0 to ensure training is tractable.
    # First sample Unif[1,2,3] random indexes from 0..T-1, weighted by the base demand values.
    base_array = np.array(base_scaled)
    base_probs = base_array / np.sum(base_array)
    # choose a random number of indexes to pick: 1, 2, or 3
    num_indexes = rng.choice([1, 2, 3])
    chosen_indexes = rng.choice(int(T), size=num_indexes, replace=False, p=base_probs)

    # select the base demand values at those indexes
    new_base_scaled = [0.0 for _ in range(int(T))]
    for i in chosen_indexes:
        new_base_scaled[i] = base_scaled[i]
    
    # rescale new_base_scaled to have the same sum as base_scaled
    sum_new_base = sum(new_base_scaled)
    sum_base = sum(base_scaled)
    if sum_new_base > 0:
        new_base_scaled = [x * (sum_base / sum_new_base) for x in new_base_scaled]
    base_scaled = new_base_scaled

    # I want roughly 2 out of the T demands to have base > 0 and flexible > 0 to ensure training is tractable.
    # First sample Unif[1,2,3] random indexes from 0..T-1, weighted by the flexible demand values.
    if proportion_base < 1.0:
        flexible_array = np.array(flexible_scaled)
        flexible_probs = flexible_array / np.sum(flexible_array)
        num_indexes = rng.choice([1, 2, 3])
        chosen_indexes = rng.choice(int(T), size=num_indexes, replace=False, p=flexible_probs)

        # select the flexible demand values at those indexes
        new_flexible_scaled = [0.0 for _ in range(int(T))]
        for i in chosen_indexes:
            new_flexible_scaled[i] = flexible_scaled[i]

        # rescale new_flexible_scaled to have the same sum as flexible_scaled
        sum_new_flexible = sum(new_flexible_scaled)
        sum_flexible = sum(flexible_scaled)
        if sum_new_flexible > 0:
            new_flexible_scaled = [x * (sum_flexible / sum_new_flexible) for x in new_flexible_scaled]
        flexible_scaled = new_flexible_scaled
    else:
        # if proportion_base == 1.0, then flexible_scaled should be all zeros
        flexible_scaled = [0.0 for _ in range(int(T))]

    # Compute per-bucket flexible deadlines within the window [0..T-1]
    # Convert hour choices to bucket counts for the current bucket_minutes
    hour_to_buckets = [int(h * 60 // bucket_minutes) for h in flexible_deadline_hours_choices]
    deltas: list[int] = [int(T - 1) for _ in range(int(T))]
    for i in range(int(T)):
        if flexible_scaled[i] > 0:
            deadline = int(rng.choice(hour_to_buckets))
            deltas[i] = min(i + deadline, int(T) - 1)

    return base_scaled, flexible_scaled, deltas, day_exploded


__all__ = [
    "prepare_exploded_from_csv",
    "build_exploded",
    "load_trace_dataframe",
    "clean_trace_dataframe",
    "add_datetime_columns",
    "compute_base_flexible_series",
    "contiguous_indexes",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Trace loader demo")
    parser.add_argument("--csv", default="batch_task.csv", help="Path to batch_task.csv")
    parser.add_argument("--bucket-minutes", type=int, default=15, help="Bucket granularity in minutes")
    parser.add_argument("--T", type=int, default=96, help="Window length in buckets (e.g., 96 for 1 day @ 15m)")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    parser.add_argument("--scale", type=float, default=40.0, help="Scale divisor")
    args = parser.parse_args()

    ctx = prepare_exploded_from_csv(args.csv, bucket_minutes=args.bucket_minutes)
    start_bucket = int(ctx["bucket_stats"]["bucket"].min())
    idxs = contiguous_indexes(start_bucket, args.T)

    base_scaled, flex_scaled, deltas, details = compute_base_flexible_series(
        ctx, day_bucket_indexes=idxs, T=args.T, seed=args.seed, scale_divisor=args.scale
    )

    print(f"Produced series of length {args.T}.")
    print(f"Base sum (scaled): {sum(base_scaled):.3f}")
    print(f"Flex sum (scaled): {sum(flex_scaled):.3f}")
    print(f"First 10 base: {[round(x,3) for x in base_scaled[:10]]}")
    print(f"First 10 flex: {[round(x,3) for x in flex_scaled[:10]]}")
    print(f"First 10 deltas: {deltas[:10]}")
