#!/usr/bin/env python3
"""Fetch a bounded historical interval into an importable replay file."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def fetch(endpoint, parameters):
    url = "https://api.openf1.org/v1/" + endpoint + "?" + urlencode(parameters)
    with urlopen(Request(url, headers={"User-Agent": "APEX-R-hackathon/1.0"}), timeout=30) as response:
        data = json.load(response)
    if not isinstance(data, list):
        raise ValueError("Unexpected OpenF1 response")
    return data, url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=int, required=True)
    parser.add_argument("--driver", type=int, required=True)
    parser.add_argument("--start", required=True, help="UTC ISO timestamp with Z or +00:00")
    parser.add_argument("--end", required=True, help="UTC ISO timestamp, no more than ten minutes after start")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--location", action="store_true", help="Include recorded location samples")
    args = parser.parse_args()
    try:
        start = datetime.fromisoformat(args.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(args.end.replace("Z", "+00:00"))
        if start.tzinfo is None or end.tzinfo is None or not 0 < (end - start).total_seconds() <= 600:
            raise ValueError("Use timezone-aware timestamps covering 0 to 600 seconds")
        if args.session <= 0 or not 1 <= args.driver <= 99:
            raise ValueError("Invalid session or driver number")
        parameters = {"session_key": args.session, "driver_number": args.driver, "date>=": start.astimezone(timezone.utc).isoformat(), "date<=": end.astimezone(timezone.utc).isoformat()}
        rows, url = fetch("car_data", parameters)
        if len(rows) < 2:
            raise ValueError("No usable telemetry in this interval. Check the session and UTC timestamps.")
        data = {"car_data": rows, "metadata": {"source": "OpenF1", "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()}}
        if args.location:
            data["location"], data["metadata"]["location_url"] = fetch("location", parameters)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(data, stream, separators=(",", ":"))
        print(f"Saved {len(rows)} car samples to {args.output}. Import this JSON from Data & model.")
    except (HTTPError, URLError, ValueError, OSError) as error:
        print(f"Historical fetch failed: {error}. Use another permitted interval or import an existing export; synthetic mode remains available.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
