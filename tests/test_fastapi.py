"""Optional FastAPI adapter and WebSocket checks after installing requirements.txt."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) for name in ("fastapi", "httpx"))


@unittest.skipUnless(AVAILABLE, "FastAPI/httpx dependencies are not installed")
class FastAPIAdapterTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from backend.app import app
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("APEX_DATABASE")
        os.environ["APEX_DATABASE"] = str(Path(self.temp.name) / "fastapi.db")
        self.client = TestClient(app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        if self.previous is None:
            os.environ.pop("APEX_DATABASE", None)
        else:
            os.environ["APEX_DATABASE"] = self.previous
        self.temp.cleanup()

    def test_comparison_and_validation_errors(self):
        self.assertTrue(self.client.get("/api/health").json()["websocket"])
        response = self.client.post("/api/compare", json={"input": {"scenarioId": "pressure"}, "request_id": "fastapi-test-123"})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["optimisation"]["recommendation"], "DEFEND")
        self.assertEqual(len(self.client.get("/api/audit").json()["records"]), 1)
        self.assertEqual(self.client.post("/api/compare", json={"input": {"state": {"soc": -1}}, "request_id": "fastapi-invalid-123"}).status_code, 422)

    def test_websocket_replays_labelled_synthetic_frames(self):
        with self.client.websocket_connect("/ws/replay") as socket:
            socket.send_json({"time": 0, "soc": 42})
            first = socket.receive_json()
            socket.send_json({"time": 5, "soc": 42})
            second = socket.receive_json()
            self.assertEqual(first["source"], "synthetic")
            self.assertEqual(first["frame"]["soc"], 42)
            self.assertGreater(second["frame"]["t"], first["frame"]["t"])
            socket.send_json({"time": -1, "soc": 42})
            self.assertIn("error", socket.receive_json())


if __name__ == "__main__":
    unittest.main()
