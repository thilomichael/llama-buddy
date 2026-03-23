"""Tests for the idle-model watchdog."""

from __future__ import annotations

from unittest.mock import patch

from llama_buddy.watchdog import _get_child_port, _is_sleeping, _unload_model, run


def _make_model(model_id: str, status: str, port: int | None = None) -> dict:
    args = ["llama-server", "--host", "127.0.0.1"]
    if port is not None:
        args.extend(["--port", str(port)])
    return {
        "id": model_id,
        "status": {"value": status, "args": args},
    }


class TestGetChildPort:
    def test_extracts_port(self):
        model = _make_model("org/model", "loaded", port=58000)
        assert _get_child_port(model) == 58000

    def test_returns_none_for_unloaded(self):
        model = _make_model("org/model", "unloaded")
        assert _get_child_port(model) is None

    def test_returns_none_for_port_zero(self):
        model = _make_model("org/model", "loaded", port=0)
        assert _get_child_port(model) is None

    def test_returns_none_for_missing_args(self):
        model = {"id": "org/model", "status": {"value": "loaded"}}
        assert _get_child_port(model) is None


class TestIsSleeping:
    def test_sleeping_true(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.get.return_value.json.return_value = {"is_sleeping": True}
            assert _is_sleeping(58000) is True
            mock_httpx.get.assert_called_once_with(
                "http://localhost:58000/props", timeout=3
            )

    def test_sleeping_false(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.get.return_value.json.return_value = {"is_sleeping": False}
            assert _is_sleeping(58000) is False

    def test_connection_error(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.get.side_effect = ConnectionError
            assert _is_sleeping(58000) is False


class TestUnloadModel:
    def test_success(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.post.return_value.json.return_value = {"success": True}
            assert _unload_model(8080, "org/model") is True
            mock_httpx.post.assert_called_once_with(
                "http://localhost:8080/models/unload",
                json={"model": "org/model"},
                timeout=10,
            )

    def test_failure(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.post.return_value.json.return_value = {"success": False}
            assert _unload_model(8080, "org/model") is False

    def test_connection_error(self):
        with patch("llama_buddy.watchdog.httpx") as mock_httpx:
            mock_httpx.post.side_effect = ConnectionError
            assert _unload_model(8080, "org/model") is False


class TestRun:
    def test_unloads_sleeping_model(self):
        models_response = {
            "data": [
                _make_model("org/awake", "loaded", port=58001),
                _make_model("org/sleeping", "loaded", port=58002),
                _make_model("org/idle", "unloaded"),
            ]
        }

        get_count = 0

        def fake_get(url, **kwargs):
            nonlocal get_count
            if "/models" in url:
                get_count += 1
                if get_count > 1:
                    raise ConnectionError("stop loop")

            class Resp:
                def json(self_):
                    if "/models" in url:
                        return models_response
                    if "58001" in url:
                        return {"is_sleeping": False}
                    if "58002" in url:
                        return {"is_sleeping": True}
                    return {}

            return Resp()

        def fake_post(url, **kwargs):
            class Resp:
                def json(self_):
                    return {"success": True}

            fake_post.calls.append((url, kwargs))
            return Resp()

        fake_post.calls = []

        with (
            patch("llama_buddy.watchdog.time.sleep"),
            patch("llama_buddy.watchdog.httpx.get", side_effect=fake_get),
            patch("llama_buddy.watchdog.httpx.post", side_effect=fake_post),
        ):
            run(8080, 30)

        assert len(fake_post.calls) == 1
        url, kwargs = fake_post.calls[0]
        assert "models/unload" in url
        assert kwargs["json"]["model"] == "org/sleeping"

    def test_exits_when_server_unreachable(self):
        call_count = 0

        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1

        with (
            patch("llama_buddy.watchdog.time.sleep", side_effect=fake_sleep),
            patch(
                "llama_buddy.watchdog.httpx.get",
                side_effect=ConnectionError,
            ),
        ):
            run(8080, 30)

        assert call_count == 1

    def test_skips_unloaded_models(self):
        models_response = {
            "data": [_make_model("org/idle", "unloaded")]
        }

        get_count = 0

        def fake_get(url, **kwargs):
            nonlocal get_count
            if "/models" in url:
                get_count += 1
                if get_count > 1:
                    raise ConnectionError("stop loop")

            class Resp:
                def json(self_):
                    return models_response

            return Resp()

        with (
            patch("llama_buddy.watchdog.time.sleep"),
            patch("llama_buddy.watchdog.httpx.get", side_effect=fake_get),
            patch("llama_buddy.watchdog.httpx.post") as mock_post,
        ):
            run(8080, 30)

        mock_post.assert_not_called()
