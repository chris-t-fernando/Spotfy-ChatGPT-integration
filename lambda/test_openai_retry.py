import json
import requests

import app


class FakeResponse:
    def __init__(self, status_code, headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = ""


def fake_request_factory(events):
    events = list(events)

    def _request():
        outcome = events.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _request


ORIGINAL_SLEEP = app.sleep_for


def test_retry_after_respected():
    sleeps = []
    app.sleep_for = lambda s: sleeps.append(s)
    try:
        responses = [
            FakeResponse(429, {"Retry-After": "1"}),
            FakeResponse(200),
        ]
        app.openai_request_with_retries(fake_request_factory(responses), max_attempts=3)
        assert len(sleeps) == 1 and 0.9 <= sleeps[0] <= 1.5
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_exponential_backoff_without_retry_after():
    sleeps = []
    app.sleep_for = lambda s: sleeps.append(s)
    try:
        responses = [
            FakeResponse(429),
            FakeResponse(200),
        ]
        app.openai_request_with_retries(fake_request_factory(responses), max_attempts=3)
        assert len(sleeps) == 1 and sleeps[0] >= 2
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_no_retry_on_400():
    app.sleep_for = lambda s: None
    try:
        responses = [FakeResponse(400)]
        resp = app.openai_request_with_retries(fake_request_factory(responses), max_attempts=2)
        assert resp.status_code == 400
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_retry_on_500():
    sleeps = []
    app.sleep_for = lambda s: sleeps.append(s)
    try:
        responses = [
            FakeResponse(500),
            FakeResponse(200),
        ]
        resp = app.openai_request_with_retries(fake_request_factory(responses), max_attempts=3)
        assert resp.status_code == 200 and len(sleeps) == 1
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_abort_when_sleep_exceeds_remaining_time():
    app.sleep_for = lambda s: None
    try:
        responses = [
            FakeResponse(429, {"Retry-After": "20"}),
        ]

        def remaining():
            return 4000  # 4 seconds remaining, insufficient for 20s sleep + margin

        try:
            app.openai_request_with_retries(
                fake_request_factory(responses),
                max_attempts=2,
                remaining_time_ms_fn=remaining,
            )
            assert False, "Expected HTTPError due to timeout budget"
        except app.HTTPError as exc:
            assert (
                exc.details.get("reason") == "openai_rate_limited_timeout_budget"
            ), exc.details
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_abort_before_request_when_no_budget():
    app.sleep_for = lambda s: None
    attempts = {"count": 0}

    def remaining():
        return 2000  # less than default min_request_budget_ms

    def request():
        attempts["count"] += 1
        return FakeResponse(200)

    try:
        try:
            app.openai_request_with_retries(
                request, max_attempts=1, remaining_time_ms_fn=remaining
            )
            assert False, "Expected HTTPError due to insufficient budget"
        except app.HTTPError as exc:
            assert attempts["count"] == 0
            assert exc.details.get("reason") == "openai_rate_limited_timeout_budget"
    finally:
        app.sleep_for = ORIGINAL_SLEEP


def test_batch_request_single_call():
    original_call = app.openai_request_with_retries
    original_get_param = app.get_secure_parameter

    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(self._payload)
                        }
                    }
                ]
            }

    payload = {
        "results": [
            {
                "playlist_id": "abc",
                "base_name": "Focus Set",
                "description": "desc",
                "tracks": [{"artist": "A", "title": "Song"}],
            }
        ]
    }

    calls = {"count": 0}

    def fake_request(callable_fn, **kwargs):
        calls["count"] += 1
        return DummyResponse(payload)

    app.openai_request_with_retries = fake_request
    app.get_secure_parameter = lambda _: "fake"
    try:
        result = app.request_batch_playlist_specs(
            [{"playlist_id": "abc", "prompt": "do thing", "track_count": 50}]
        )
        assert calls["count"] == 1
        assert "abc" in result and result["abc"]["base_name"] == "Focus Set"
    finally:
        app.openai_request_with_retries = original_call
        app.get_secure_parameter = original_get_param


def test_batch_request_missing_ids_ignored():
    original_call = app.openai_request_with_retries
    original_get_param = app.get_secure_parameter

    class DummyResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(self._payload)
                        }
                    }
                ]
            }

    payload = {"results": [{"base_name": "x", "description": "", "tracks": []}]}

    def fake_request(callable_fn, **kwargs):
        return DummyResponse(payload)

    app.openai_request_with_retries = fake_request
    app.get_secure_parameter = lambda _: "fake"
    try:
        result = app.request_batch_playlist_specs(
            [{"playlist_id": "abc", "prompt": "do thing", "track_count": 50}]
        )
        assert result == {}
    finally:
        app.openai_request_with_retries = original_call
        app.get_secure_parameter = original_get_param


if __name__ == "__main__":
    test_retry_after_respected()
    test_exponential_backoff_without_retry_after()
    test_no_retry_on_400()
    test_retry_on_500()
    test_abort_when_sleep_exceeds_remaining_time()
    test_abort_before_request_when_no_budget()
    test_batch_request_single_call()
    test_batch_request_missing_ids_ignored()
    print("openai_retry tests passed")
