"""HTTP contract and persistent storage tests. No third-party packages required."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.standalone import create_server
from backend.service import Store


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.database = Path(cls.temp.name) / "test.db"
        cls.server = create_server(0, cls.database)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()

    def request(self, path, data=None, headers=None):
        request = Request(self.base + path, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def test_static_assets_and_health(self):
        for path in ("/", "/engine.js", "/telemetry.js", "/app.js", "/styles.css", "/worker.js", "/favicon.svg"):
            status, body = self.request(path)
            self.assertEqual(status, 200, path)
            self.assertGreater(len(body), 40)
        status, body = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["engine_version"], "1.0.0")

    def test_comparison_persists_and_duplicate_request_is_idempotent(self):
        payload = {"input": {"scenarioId": "opportunity", "seed": 78}, "provenance": {"replay": "synthetic"}, "request_id": "test-decision-12345"}
        status, first = self.request("/api/compare", payload)
        self.assertEqual(status, 200, first)
        status, second = self.request("/api/compare", payload)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(first), json.loads(second))
        self.assertEqual(len(Store(self.database).recent()), 1)
        self.assertEqual(json.loads(first)["apex"]["summary"]["executedViolations"], 0)
        payload["input"]["seed"] = 79
        self.assertEqual(self.request("/api/compare", payload)[0], 422)

    def test_invalid_payloads_and_origin_are_rejected(self):
        for bad in ({"state": {"soc": -1}}, {"horizon": 9}, {"judgeAction": "INVALID"}):
            status, _ = self.request("/api/compare", {"input": bad, "request_id": "invalid-12345"})
            self.assertEqual(status, 422)
        self.assertEqual(self.request("/api/health", headers={"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.request("/api/audit?limit=101")[0], 422)
        self.assertEqual(self.request("/api/missing")[0], 404)
        self.assertEqual(self.request("/../backend/service.py")[0], 404)

    def test_benchmark_and_replay_contract(self):
        status, body = self.request("/api/validate", {"count": 5, "seed": 42})
        self.assertEqual(status, 200)
        value = json.loads(body)
        self.assertEqual(value["count"], value["wins"] + value["ties"] + value["losses"])
        status, body = self.request("/api/telemetry?soc=35")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["frames"][0]["soc"], 35)
        self.assertEqual(self.request("/api/telemetry?soc=101")[0], 422)


if __name__ == "__main__":
    unittest.main()
