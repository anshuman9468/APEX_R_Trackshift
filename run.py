#!/usr/bin/env python3
"""Run the full local app. Use FastAPI when installed, otherwise the stdlib adapter."""

import argparse
import importlib.util
import sys


def main():
    parser = argparse.ArgumentParser(description="APEX-R local race strategy application")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--standalone", action="store_true", help="Use the zero-dependency HTTP adapter")
    parser.add_argument("--fastapi", action="store_true", help="Require FastAPI rather than falling back")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be 1..65535")
    installed = all(importlib.util.find_spec(m) for m in ("fastapi", "uvicorn"))
    if args.fastapi and not installed:
        parser.error("Install requirements.txt in a virtual environment first, or omit --fastapi")
    if installed and not args.standalone:
        import uvicorn
        print(f"APEX-R: http://127.0.0.1:{args.port}/ | API docs: /docs", flush=True)
        uvicorn.run("backend.app:app", host="127.0.0.1", port=args.port, log_level="warning")
    else:
        from backend.standalone import create_server
        try:
            server = create_server(args.port)
        except (OSError, RuntimeError) as error:
            print(f"Cannot start APEX-R: {error}. Try --port 8001 if the port is occupied.", file=sys.stderr)
            return 1
        print(f"APEX-R: http://127.0.0.1:{args.port}/ | Local HTTP API + SQLite | Ctrl+C to stop", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
