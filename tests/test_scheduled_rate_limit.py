import json
import os
import sys
import time

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "lambda"))
import app  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload or {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"results": []})
                    }
                }
            ]
        }
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeTable:
    def __init__(self, items):
        self.items = items
        self.update_calls = []

    def scan(self, **kwargs):
        return {"Items": list(self.items), "LastEvaluatedKey": None}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)


class FakeDDBResource:
    def __init__(self, table):
        self._table = table

    def Table(self, name):
        return self._table


@pytest.fixture(autouse=True)
def reset_transport(monkeypatch):
    monkeypatch.setattr(app, "OPENAI_TRANSPORT", None)


def base_config():
    config = dict(app.ENV_CONFIG)
    config["openai_throttle_ms"] = 0
    config["scheduled_openai_max_attempts"] = 1
    return config


def test_scheduled_429_deferred_no_retry(monkeypatch):
    items = [
        {
            "playlist_id": "pl1",
            "base_prompt": "Curate focus tracks",
            "history_entries": [],
            "enabled": True,
        }
    ]
    table = FakeTable(items)
    monkeypatch.setattr(app, "DDB_RESOURCE", FakeDDBResource(table))
    monkeypatch.setattr(app, "ENV_CONFIG", base_config())
    monkeypatch.setattr(app, "get_secure_parameter", lambda _: "fake-key")
    monkeypatch.setattr(app, "sleep_for", lambda s: (_ for _ in ()).throw(AssertionError("sleep not expected")))
    monkeypatch.setattr(app, "refresh_spotify_access_token", lambda: (_ for _ in ()).throw(AssertionError("spotify not expected")))
    calls = {"count": 0}

    def fake_transport(payload):
        calls["count"] += 1
        return FakeResponse(429, headers={"Retry-After": "60"})

    monkeypatch.setattr(app, "OPENAI_TRANSPORT", fake_transport)
    monkeypatch.setattr(app.random, "randint", lambda a, b: 0)

    summary = app.process_scheduled_event()

    assert calls["count"] == 1
    assert summary["deferred"] == 1
    assert summary["results"][0]["outcome"] == "deferred"
    assert summary["results"][0]["error_category"] == "openai_rate_limited"
    assert summary["results"][0]["retry_after_s"] == 60
    next_epoch = summary["results"][0]["next_eligible_at_epoch"]
    assert next_epoch is not None
    assert table.update_calls
    stored_epoch = int(table.update_calls[0]["ExpressionAttributeValues"][":ts"])
    assert stored_epoch == next_epoch
    assert 55 <= next_epoch - int(time.time()) <= 65


def test_scheduled_skips_within_backoff(monkeypatch):
    future_epoch = int(time.time()) + 120
    items = [
        {
            "playlist_id": "pl2",
            "base_prompt": "Upbeat mix",
            "history_entries": [],
            "enabled": True,
            "next_eligible_at_epoch": future_epoch,
        }
    ]
    table = FakeTable(items)
    monkeypatch.setattr(app, "DDB_RESOURCE", FakeDDBResource(table))
    monkeypatch.setattr(app, "ENV_CONFIG", base_config())
    monkeypatch.setattr(app, "get_secure_parameter", lambda _: "fake-key")

    def fail_transport(payload):
        raise AssertionError("OpenAI should not be called during backoff")

    monkeypatch.setattr(app, "OPENAI_TRANSPORT", fail_transport)

    summary = app.process_scheduled_event()

    assert summary["skipped"] == 1
    assert summary["results"][0]["outcome"] == "skipped"
    assert summary["results"][0]["error_category"] == "rate_limit_backoff"
    assert summary["results"][0]["next_eligible_at_epoch"] == future_epoch
    assert not table.update_calls
