"""Focused HTTP/inference checks for the frozen GNN backend integration."""

from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from backend.gnn_adapter import FrozenGNNAdapter, OUTPUT_LABEL
from backend.standalone import create_server


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "gnn_predict_request.json"


class FrozenGNNBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_device = os.environ.get("APEX_GNN_DEVICE")
        os.environ["APEX_GNN_DEVICE"] = "cpu"
        cls.temp = tempfile.TemporaryDirectory(prefix="apexr-gnn-http-")
        cls.server = create_server(0, Path(cls.temp.name) / "test.db")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        cls.example = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.temp.cleanup()
        if cls.previous_device is None:
            os.environ.pop("APEX_GNN_DEVICE", None)
        else:
            os.environ["APEX_GNN_DEVICE"] = cls.previous_device

    def request(self, path, data=None):
        request = Request(
            self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            return error.code, json.loads(error.read())

    def test_status_loads_frozen_checkpoint(self):
        status, body = self.request("/api/model/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "available")
        self.assertTrue(body["loaded"])
        self.assertEqual(body["selected_seed"], 42)
        self.assertEqual(body["selected_epoch"], 28)
        self.assertEqual(body["output_label"], OUTPUT_LABEL)
        self.assertEqual(body["checkpoint_sha256"], "d5ce7258fa2ec62f953d516816497f1cdbd04b1b839e362ef447f79c4e618d14")

    def test_real_http_prediction_and_stability(self):
        first_status, first = self.request("/api/model/predict", self.example)
        second_status, second = self.request("/api/model/predict", self.example)
        self.assertEqual(first_status, 200, first)
        self.assertEqual(second_status, 200, second)
        self.assertEqual(first["status"], "available")
        self.assertEqual(first["score_label"], OUTPUT_LABEL)
        self.assertTrue(math.isfinite(first["proxy_score"]))
        self.assertGreaterEqual(first["proxy_score"], 0.0)
        self.assertLessEqual(first["proxy_score"], 1.0)
        self.assertEqual(first["proxy_score"], second["proxy_score"])
        self.assertIsNone(first["strategy_action"])
        self.assertGreaterEqual(first["inference_latency_ms"], 0.0)

    def test_reference_parity_for_approved_fixture(self):
        status, body = self.request("/api/model/predict", self.example)
        self.assertEqual(status, 200)
        self.assertAlmostEqual(body["proxy_score"], 0.0052168904803693295, delta=1e-7)

    def test_rejects_labels_bad_shape_and_protected_session(self):
        with_label = copy.deepcopy(self.example)
        with_label["graph"]["label"] = 0
        self.assertEqual(self.request("/api/model/predict", with_label)[0], 422)

        bad_shape = copy.deepcopy(self.example)
        bad_shape["graph"]["pair_raw"] = [0.0]
        self.assertEqual(self.request("/api/model/predict", bad_shape)[0], 422)

        protected = copy.deepcopy(self.example)
        protected["graph"]["race_id"] = "11353"
        status, body = self.request("/api/model/predict", protected)
        self.assertEqual(status, 422)
        self.assertIn("Protected session", body["detail"])

    def test_existing_routes_still_work(self):
        self.assertEqual(self.request("/api/health")[0], 200)
        self.assertEqual(self.request("/api/phase5/metadata")[0], 200)


if __name__ == "__main__":
    unittest.main()
