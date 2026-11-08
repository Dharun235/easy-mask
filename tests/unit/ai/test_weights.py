"""The weights manager against a real HTTP server on localhost (no internet needed)."""

import hashlib
import http.server
import socket
import threading
from pathlib import Path

import pytest

from easy_mask.ai import weights
from easy_mask.ai.weights import WeightFile, WeightsError

PAYLOAD = b"fake model weights " * 5000  # ~95 kB: several chunks with a small CHUNK


class Handler(http.server.BaseHTTPRequestHandler):
    """Serves PAYLOAD at /model.onnx; /truncated.onnx promises more bytes than it sends."""

    def do_GET(self):
        if self.path == "/missing.onnx":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(PAYLOAD)))
        self.end_headers()
        if self.path == "/truncated.onnx":
            self.wfile.write(PAYLOAD[: len(PAYLOAD) // 2])
            self.close_connection = True
        else:
            self.wfile.write(PAYLOAD)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(weights, "CHUNK", 16 * 1024)


def spec(url, payload=PAYLOAD):
    return WeightFile(url, hashlib.sha256(payload).hexdigest(), len(payload))


def test_download_verifies_and_reports_progress(server, tmp_path):
    seen = []
    path = weights.ensure(spec(f"{server}/model.onnx"), tmp_path, lambda *a: seen.append(a))
    assert path == tmp_path / "model.onnx" and path.read_bytes() == PAYLOAD
    assert seen[-1] == ("model.onnx", len(PAYLOAD), len(PAYLOAD))
    assert len(seen) > 1 and [done for _, done, _ in seen] == sorted(done for _, done, _ in seen)
    assert list(tmp_path.iterdir()) == [path]  # no .part left behind


def test_cached_file_is_not_downloaded_again(tmp_path):
    (tmp_path / "model.onnx").write_bytes(PAYLOAD)
    offline = spec("http://127.0.0.1:9/model.onnx")
    assert weights.cached_path(offline, tmp_path) == tmp_path / "model.onnx"
    assert weights.ensure(offline, tmp_path) == tmp_path / "model.onnx"


def test_wrong_size_cached_file_is_replaced(server, tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"old")
    assert weights.cached_path(spec(f"{server}/model.onnx"), tmp_path) is None
    assert weights.ensure(spec(f"{server}/model.onnx"), tmp_path).read_bytes() == PAYLOAD


def test_checksum_mismatch_is_rejected(server, tmp_path):
    bad = WeightFile(f"{server}/model.onnx", "0" * 64, len(PAYLOAD))
    with pytest.raises(WeightsError, match="corrupt"):
        weights.ensure(bad, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_truncated_download_is_rejected_and_cleaned_up(server, tmp_path):
    with pytest.raises(WeightsError):
        weights.ensure(spec(f"{server}/truncated.onnx"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_stale_partial_file_is_overwritten(server, tmp_path):
    (tmp_path / "model.onnx.part").write_bytes(b"left over from a crash")
    assert weights.ensure(spec(f"{server}/model.onnx"), tmp_path).read_bytes() == PAYLOAD
    assert not (tmp_path / "model.onnx.part").exists()


def test_http_error_is_reported(server, tmp_path):
    with pytest.raises(WeightsError, match="404"):
        weights.ensure(spec(f"{server}/missing.onnx"), tmp_path)


def test_offline_gives_a_helpful_error(tmp_path):
    with socket.socket() as sock:  # a port that nothing listens on
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(WeightsError, match="internet connection"):
        weights.ensure(spec(f"http://127.0.0.1:{port}/model.onnx"), tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_cache_dir_defaults_and_overrides(monkeypatch, tmp_path):
    monkeypatch.delenv("EASY_MASK_MODELS", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert weights.cache_dir() == Path.home() / ".cache" / "easy-mask" / "models"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert weights.cache_dir() == tmp_path / "easy-mask" / "models"
    monkeypatch.setenv("EASY_MASK_MODELS", str(tmp_path / "m"))
    assert weights.cache_dir() == tmp_path / "m"


def test_default_folder_is_the_cache_dir(server, monkeypatch, tmp_path):
    monkeypatch.setenv("EASY_MASK_MODELS", str(tmp_path / "models"))
    path = weights.ensure(spec(f"{server}/model.onnx"))
    assert path == tmp_path / "models" / "model.onnx"
