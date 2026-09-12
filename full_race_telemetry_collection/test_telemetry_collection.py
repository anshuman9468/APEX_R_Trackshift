#!/usr/bin/env python3
"""Compatibility entry point for the independent collection checks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from verify_telemetry_collection import main as verify_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args, _ = parser.parse_known_args()
    sys.argv = [sys.argv[0], "--root", str(args.root), "--workspace", str(args.workspace)]
    raise SystemExit(verify_main())
