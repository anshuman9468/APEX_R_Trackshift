#!/usr/bin/env python3
"""Build compact circuit layouts from public FastF1 XY telemetry.

The live MultiViewer timing API identifies the meeting/circuit but does not
expose circuit coordinates.  This one-time build step turns a clean fastest
lap from each requested season/event into a normalized SVG point sequence.
The browser never downloads telemetry; it only reads the generated catalogue.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fastf1
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "frontend" / "track_maps.js"


def _event_name(event) -> str:
    return str(event.get("EventName", "")).strip()


def _normalise_xy(telemetry) -> list[list[float]]:
    xy = telemetry[["X", "Y"]].dropna().to_numpy(dtype=float)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 20:
        raise ValueError("fastest lap did not contain enough finite X/Y samples")
    # Remove repeated samples, preserve the order around the racing line, and
    # keep a compact but visibly faithful shape for the SVG viewport.
    xy = xy[np.concatenate(([True], np.any(np.diff(xy, axis=0) != 0, axis=1)))]
    target = 180
    indices = np.linspace(0, len(xy) - 1, min(target, len(xy)), dtype=int)
    xy = xy[indices]
    xy = np.vstack([xy, xy[0]])
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    width, height = max(1.0, xmax - xmin), max(1.0, ymax - ymin)
    scale = min(700.0 / width, 360.0 / height)
    offset_x = 400.0 - ((xmin + xmax) / 2.0) * scale
    offset_y = ((ymin + ymax) / 2.0) * scale
    points = [[round(float(x * scale + offset_x), 3), round(float(220.0 - (y * scale - offset_y)), 3)] for x, y in xy]
    return points


def _load_map(year: int, event_name: str, session_name: str) -> dict:
    session = fastf1.get_session(year, event_name, session_name)
    session.load(telemetry=True, laps=True, weather=False, messages=False)
    fastest = session.laps.pick_fastest()
    if fastest is None or not getattr(fastest, "Driver", None):
        raise ValueError("no fastest lap available")
    telemetry = fastest.get_telemetry()
    return {
        "event": str(session.event.EventName),
        "season": year,
        "location": str(session.event.Location),
        "official_name": str(session.event.OfficialEventName),
        "session": session_name,
        "driver": str(fastest.Driver),
        "points": _normalise_xy(telemetry),
        "source": f"FastF1 {year} {session.event.EventName} {session_name} fastest-lap XY telemetry",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--session", default="R", help="FastF1 session identifier, usually R")
    parser.add_argument("--cache", type=Path, default=ROOT / "runtime" / "fastf1_cache")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {args.output}; pass --overwrite to replace it")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(args.cache)
    maps = {}
    failures = []
    for year in sorted(set(args.years)):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            event_name = _event_name(event)
            if not event_name:
                continue
            try:
                record = _load_map(year, event_name, args.session)
                for key in {record["event"], record["location"]}:
                    maps[f"{year}:{key}".lower()] = record
                print(f"mapped {year} {record['event']} ({len(record['points'])} points)", flush=True)
            except Exception as error:  # keep one unavailable event from stopping the calendar
                failures.append({"year": year, "event": event_name, "error": str(error)})
                print(f"skipped {year} {event_name}: {error}", flush=True)
    payload = {
        "version": 1,
        "coordinate_system": "normalized FastF1 fastest-lap XY telemetry, SVG viewBox 0 0 800 440",
        "maps": maps,
        "failures": failures,
    }
    javascript = "window.ApexTrackMaps = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    args.output.write_text(javascript, encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(args.output), "maps": len(maps), "failures": len(failures)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
