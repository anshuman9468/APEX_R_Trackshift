#!/usr/bin/env python3
"""Add candidate-specific telemetry counts to the passed-race event review.

This is an offline audit helper. It reads only the already collected PASS-race
streams, uses the laptime boundary session times to form conservative audit
intervals, and never promotes a candidate to a verified overtake.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path


def parse_num(value):
    try:
        return float(value) if str(value).strip() else None
    except (TypeError, ValueError):
        return None


def i(value):
    x = parse_num(value)
    return int(x) if x is not None and x.is_integer() else None


def key(row):
    return f"{i(row.get('year'))}:{row.get('event','').strip()}:{row.get('session','').strip()}"


def load_laptimes(root: Path, out: Path, pass_ids: set[str]):
    result = {}
    with (out / "laptime_source_inventory_37.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rid = key(r)
            if rid not in pass_ids:
                continue
            cache = Path(r.get("cache_path", ""))
            if not cache.is_file():
                raise FileNotFoundError(cache)
            data = json.loads(cache.read_text(encoding="utf-8"))
            by_lap = {}
            laps = data.get("lap", [])
            for n, lap in enumerate(laps):
                lap = i(lap)
                if lap is None:
                    continue
                get = lambda name: data.get(name, [None] * len(laps))[n] if n < len(data.get(name, [])) else None
                by_lap[lap] = {"start": parse_num(get("lST")), "end": parse_num(get("sesT"))}
            result[(rid, r.get("driver", "").strip())] = by_lap
    return result


def make_intervals(candidates, lap):
    intervals = defaultdict(list)
    records = {}
    for c in candidates:
        rid = c["race_id"]; a = c["attacker"]; t = c["target_driver"]
        dl = i(c.get("decision_lap")); el = i(c.get("endpoint_lap"))
        a0, t0 = lap.get((rid, a), {}).get(dl, {}), lap.get((rid, t), {}).get(dl, {})
        a1, t1 = lap.get((rid, a), {}).get(el, {}), lap.get((rid, t), {}).get(el, {})
        decisions = [x for x in (a0.get("start"), t0.get("start")) if x is not None]
        endpoints = [x for x in (a1.get("start"), t1.get("start")) if x is not None]
        if not decisions or not endpoints:
            continue
        start, end = max(decisions), min(endpoints)
        if start > end:
            start, end = min(decisions), max(endpoints)
        cid = c["candidate_id"]
        records[cid] = {"start": start, "end": end, "attacker": a, "target": t,
                        "car_a": 0, "car_t": 0, "pos_a": 0, "pos_t": 0}
        intervals[(rid, a)].append((start, end, cid, "attacker"))
        intervals[(rid, t)].append((start, end, cid, "target"))
    for k in intervals:
        intervals[k].sort(key=lambda x: (x[0], x[1], x[2]))
    return intervals, records


def count_stream(path: Path, pass_ids: set[str], intervals, records, stream: str):
    pointers = defaultdict(int)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rid = row.get("race_id", ""); driver = row.get("driver", "")
            if rid not in pass_ids:
                continue
            t = parse_num(row.get("session_time_sec"))
            if t is None:
                continue
            k = (rid, driver); values = intervals.get(k, [])
            p = pointers[k]
            while p < len(values) and values[p][1] < t:
                p += 1
            pointers[k] = p
            j = p
            while j < len(values) and values[j][0] <= t:
                start, end, cid, role = values[j]
                if t <= end:
                    records[cid][f"{stream}_{'a' if role == 'attacker' else 't'}"] += 1
                j += 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = p.parse_args()
    root, out = args.root.resolve(), args.output_dir.resolve()
    status = list(csv.DictReader((root / "full_race_telemetry_collection_final/race_collection_status.csv").open(encoding="utf-8", newline="")))
    pass_ids = {r["race_id"] for r in status if r["telemetry_coverage_status"] == "PASS"}
    with gzip.open(out / "position_swap_candidates_37.csv.gz", "rt", encoding="utf-8", newline="") as f:
        candidates = list(csv.DictReader(f))
    lap = load_laptimes(root, out, pass_ids)
    intervals, records = make_intervals(candidates, lap)
    count_stream(root / "full_race_telemetry_collection_final/telemetry_car.csv.gz", pass_ids, intervals, records, "car")
    count_stream(root / "full_race_telemetry_collection_final/telemetry_position.csv.gz", pass_ids, intervals, records, "pos")
    review_path = out / "event_review_37.csv"
    rows = list(csv.DictReader(review_path.open(encoding="utf-8", newline="")))
    fields = list(rows[0]) + ["session_interval_start_sec", "session_interval_end_sec", "candidate_interval_duration_sec", "candidate_large_interval_flag", "car_samples_attacker", "car_samples_target", "position_samples_attacker", "position_samples_target", "candidate_telemetry_status"]
    for row in rows:
        rec = records.get(row["candidate_id"])
        if not rec:
            row.update({f: "" for f in fields if f not in row})
            row["candidate_telemetry_status"] = "BOUNDARY_SESSION_TIME_UNAVAILABLE"
            continue
        row["session_interval_start_sec"] = rec["start"]; row["session_interval_end_sec"] = rec["end"]
        row["candidate_interval_duration_sec"] = round(rec["end"] - rec["start"], 6)
        row["candidate_large_interval_flag"] = "TRUE" if rec["end"] - rec["start"] > 600 else "FALSE"
        row["car_samples_attacker"] = rec["car_a"]; row["car_samples_target"] = rec["car_t"]
        row["position_samples_attacker"] = rec["pos_a"]; row["position_samples_target"] = rec["pos_t"]
        if rec["end"] - rec["start"] > 600:
            row["candidate_telemetry_status"] = "PAIR_STREAM_INTERVAL_HAS_LONG_DISCONTINUITY"
        else:
            row["candidate_telemetry_status"] = "BOTH_STREAMS_HAVE_TIMESTAMPED_SAMPLES" if min(rec["car_a"], rec["car_t"], rec["pos_a"], rec["pos_t"]) > 0 else "PAIR_STREAM_SAMPLE_GAP"
    with review_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"candidates": len(rows), "intervals": len(records), "both_streams": sum(r["candidate_telemetry_status"] == "BOTH_STREAMS_HAVE_TIMESTAMPED_SAMPLES" for r in rows), "long_discontinuities": sum(r["candidate_telemetry_status"] == "PAIR_STREAM_INTERVAL_HAS_LONG_DISCONTINUITY" for r in rows), "pair_stream_gaps": sum(r["candidate_telemetry_status"] == "PAIR_STREAM_SAMPLE_GAP" for r in rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
