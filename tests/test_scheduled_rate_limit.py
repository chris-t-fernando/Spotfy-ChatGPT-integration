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
    def __init__(self, scan_items=None, item_map=None):
        self.scan_items = scan_items or []
        self.item_map = item_map or {}
        self.update_calls = []
        self.put_calls = []
        self.delete_calls = []

    def scan(self, **kwargs):
        return {"Items": list(self.scan_items), "LastEvaluatedKey": None}

    def get_item(self, **kwargs):
        key = kwargs.get("Key", {}).get("playlist_id")
        item = self.item_map.get(key)
        return {"Item": item} if item else {}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        key = kwargs.get("Key", {}).get("playlist_id")
        value = kwargs.get("ExpressionAttributeValues", {}).get(":ts")
        if key and value is not None:
            self.item_map[key] = {"next_eligible_at_epoch": int(value)}

    def put_item(self, Item):
        self.put_calls.append(Item)
        key = Item.get("playlist_id")
        if key is not None:
            self.item_map[key] = Item

    def delete_item(self, Key):
        self.delete_calls.append(Key)
        key = Key.get("playlist_id")
        if key in self.item_map:
            del self.item_map[key]


class FakeDDBResource:
    def __init__(self, table):
        self._table = table

    def Table(self, name):
        return self._table


class FakeContext:
    function_name = "test-func"
    function_version = "$LATEST"
    aws_request_id = "req-123"
    invoked_function_arn = "arn:aws:lambda:region:acct:function:test"

    def get_remaining_time_in_millis(self):
        return 300000


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


def test_ensure_playlist_id_creates_when_blank(monkeypatch):
    item = {
        "playlist_id": "to_be_created#lofi",
        "base_prompt": "Focus vibes",
        "base_name": "Auto Focus",
        "history_entries": [],
        "enabled": True,
    }
    table = FakeTable()
    monkeypatch.setattr(app, "refresh_spotify_access_token", lambda: "token")
    monkeypatch.setattr(app, "fetch_spotify_user", lambda _: {"id": "user123"})
    monkeypatch.setattr(
        app,
        "create_playlist",
        lambda access_token, user_id, name, description: {
            "id": "new123",
            "name": name,
        },
    )

    new_id = app.ensure_playlist_id(table, item, "")
    assert new_id == "new123"
    assert item["playlist_id"] == "new123"
    assert table.put_calls and table.put_calls[0]["playlist_id"] == "new123"


def make_http_event():
    return {
        "requestContext": {"http": {"method": "POST", "path": "/playlist"}},
        "headers": {
            "x-api-key": "test-key",
            "content-type": "application/json",
        },
        "body": json.dumps({"prompt": "hi"}),
    }


def test_http_backoff_skips_openai(monkeypatch):
    future_epoch = int(time.time()) + 120
    table = FakeTable(
        scan_items=[],
        item_map={app.HTTP_BACKOFF_KEY: {"next_eligible_at_epoch": future_epoch}},
    )
    monkeypatch.setattr(app, "DDB_RESOURCE", FakeDDBResource(table))
    monkeypatch.setattr(app, "ENV_CONFIG", base_config())
    monkeypatch.setattr(app, "get_secure_parameter", lambda _: "test-key")

    def fail_transport(payload):
        raise AssertionError("OpenAI should not be called during HTTP backoff")

    monkeypatch.setattr(app, "OPENAI_TRANSPORT", fail_transport)

    response = app.handler(make_http_event(), FakeContext())
    assert response["statusCode"] == 429
    body = json.loads(response["body"])
    assert body["reason"] == "rate_limit_backoff"
    assert body["next_eligible_at_epoch"] == future_epoch
    assert "Retry-After" in response["headers"]
    assert table.update_calls == []


def test_http_openai_429_sets_backoff(monkeypatch):
    table = FakeTable()
    monkeypatch.setattr(app, "DDB_RESOURCE", FakeDDBResource(table))
    monkeypatch.setattr(app, "ENV_CONFIG", base_config())
    monkeypatch.setattr(app, "get_secure_parameter", lambda _: "test-key")
    monkeypatch.setattr(app, "sleep_for", lambda s: (_ for _ in ()).throw(AssertionError("sleep not expected")))
    monkeypatch.setattr(app, "refresh_spotify_access_token", lambda: (_ for _ in ()).throw(AssertionError("spotify not expected")))
    calls = {"count": 0}

    def fake_transport(payload):
        calls["count"] += 1
        return FakeResponse(429, headers={"Retry-After": "30"})

    monkeypatch.setattr(app, "OPENAI_TRANSPORT", fake_transport)
    monkeypatch.setattr(app.random, "randint", lambda a, b: 0)

    response = app.handler(make_http_event(), FakeContext())
    assert calls["count"] == 1
    assert response["statusCode"] == 429
    body = json.loads(response["body"])
    assert body["reason"] == "rate_limit_backoff"
    assert table.update_calls
    assert table.update_calls[0]["Key"]["playlist_id"] == app.HTTP_BACKOFF_KEY
    assert "Retry-After" in response["headers"]


def test_advance_rotation_cursor(monkeypatch):
    table = FakeTable()
    table.item_map["pl3"] = {"rotation_cursor": 0}
    app.advance_rotation_cursor(table, "pl3", 1, 3)
    assert table.update_calls
    assert table.update_calls[0]["ExpressionAttributeValues"][":cursor"] == 2
