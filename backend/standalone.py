"""Dependency-free local server using the same services as FastAPI."""

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .service import MAX_BODY, PUBLIC, Service


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, service, **kwargs):
        self.service = service
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def respond(self, data, status=200):
        body = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def allowed(self):
        host = self.headers.get("Host", "").split(":")[0]
        if host not in {"127.0.0.1", "localhost"}:
            self.respond({"detail": "Use the local server address"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc != self.headers.get("Host"):
            self.respond({"detail": "Cross-origin requests are not accepted"}, 403)
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        route = urlsplit(self.path)
        try:
            if route.path == "/api/health":
                return self.respond(self.service.health())
            if route.path == "/api/scenarios":
                return self.respond(self.service.metadata)
            if route.path == "/api/audit":
                limit = int(parse_qs(route.query).get("limit", ["100"])[0])
                return self.respond(self.service.audit(limit))
            if route.path == "/api/telemetry":
                soc = float(parse_qs(route.query).get("soc", ["42"])[0])
                return self.respond(self.service.execute("telemetry", {"soc": soc}))
            if route.path == "/api/model/status":
                return self.respond(self.service.gnn.manifest_status())
            if route.path == "/api/phase5/metadata":
                return self.respond(self.service.phase5.metadata())
            if route.path == "/api/phase5/replay":
                return self.respond(self.service.phase5.replay())
            if route.path == "/api/phase5/state":
                value = float(parse_qs(route.query).get("time", ["0"])[0])
                return self.respond(self.service.phase5.state(value))
            if route.path.startswith("/api/phase5/historical-state/"):
                window_id = route.path.rsplit("/", 1)[-1]
                return self.respond(self.service.phase5.historical_state(window_id))
            if route.path.startswith("/api/"):
                return self.respond({"detail": "Unknown endpoint"}, 404)
            if route.path.endswith("/") and route.path != "/":
                return self.respond({"detail": "Not found"}, 404)
            return super().do_GET()
        except (ValueError, TypeError) as error:
            self.respond({"detail": str(error)}, 422)
        except RuntimeError as error:
            self.respond({"detail": str(error)}, 503)

    def do_POST(self):
        if not self.allowed():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                return self.respond({"detail": "Invalid request size"}, 413)
            if self.headers.get_content_type() != "application/json":
                return self.respond({"detail": "Use application/json"}, 415)
            body = json.loads(self.rfile.read(length))
            path = urlsplit(self.path).path
            if path == "/api/compare":
                return self.respond(self.service.compare(body))
            if path == "/api/validate":
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                return self.respond(self.service.execute("validate", body))
            if path == "/api/model/predict":
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                result = self.service.gnn.predict_request(body)
                return self.respond(result, 503 if result.get("status") == "unavailable" else 200)
            if path == "/api/phase5/inference":
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                return self.respond(self.service.phase5.inference(body.get("window_id")))
            return self.respond({"detail": "Unknown endpoint"}, 404)
        except (ValueError, TypeError) as error:
            self.respond({"detail": str(error)}, 422)
        except RuntimeError as error:
            self.respond({"detail": str(error)}, 503)

    def list_directory(self, path):
        self.send_error(404)
        return None


def create_server(port=8000, database=None):
    service = Service(database)
    return ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, service=service))
