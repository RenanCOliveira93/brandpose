"""Dependency-free experimental aggregation of estimated visual attention.

Synthetic/offline records only. NOT a gaze model and NOT production code.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    """One gaze-to-zone observation from an upstream estimator.

    zone=None means unknown/out-of-frame/person/otherwise invalid.
    confidence is a model score, not a calibrated probability.
    """
    track_id: str
    timestamp_s: float
    zone: str | None
    confidence: float


@dataclass(frozen=True)
class ZoneSummary:
    zone: str
    estimated_attention_s: float
    unique_tracks: int
    accepted_observations: int


def summarize_attention(
    observations: Iterable[Observation],
    *,
    sample_interval_s: float,
    min_confidence: float = 0.5,
    max_gap_factor: float = 1.5,
) -> list[ZoneSummary]:
    """Aggregate zone attention from samples without inventing unseen dwell.

    Each accepted sample represents at most one sampling interval.
    A duplicate timestamp for the same track is ignored (first wins).
    Intervals after large observation gaps are not backfilled.
    Unknown/low-confidence observations are abstentions, not zone hits.

    This is a descriptive estimate; it does not establish true eye fixation.
    """
    import math

    if not math.isfinite(sample_interval_s) or sample_interval_s <= 0:
        raise ValueError("sample_interval_s must be positive and finite")
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    if not math.isfinite(max_gap_factor) or max_gap_factor < 1:
        raise ValueError("max_gap_factor must be finite and >= 1")

    by_track: dict[str, list[Observation]] = defaultdict(list)
    for obs in observations:
        if not obs.track_id or not math.isfinite(obs.timestamp_s):
            raise ValueError("track_id and timestamp_s must be valid")
        if not math.isfinite(obs.confidence) or not 0 <= obs.confidence <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        by_track[obs.track_id].append(obs)

    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    people: dict[str, set[str]] = defaultdict(set)

    for track_id, records in by_track.items():
        seen: set[float] = set()
        previous_t: float | None = None
        for obs in sorted(records, key=lambda r: r.timestamp_s):
            if obs.timestamp_s in seen:
                continue
            seen.add(obs.timestamp_s)
            if previous_t is None:
                duration = sample_interval_s
            else:
                gap = obs.timestamp_s - previous_t
                duration = min(gap, sample_interval_s) if gap <= max_gap_factor * sample_interval_s else 0.0
            previous_t = obs.timestamp_s
            if not obs.zone or obs.confidence < min_confidence:
                continue
            totals[obs.zone] += duration
            counts[obs.zone] += 1
            people[obs.zone].add(track_id)

    return [
        ZoneSummary(
            zone=zone,
            estimated_attention_s=round(total, 6),
            unique_tracks=len(people[zone]),
            accepted_observations=counts[zone],
        )
        for zone, total in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
    ]
