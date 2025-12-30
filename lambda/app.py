import base64
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    import boto3
    from boto3.dynamodb.conditions import Attr
except ImportError:  # pragma: no cover - executed when boto3 unavailable locally
    boto3 = None

    class Attr:  # type: ignore[override]
        """Fallback Attr stub to keep tests working without boto3."""

        def __init__(self, name: str):
            self.name = name

        def eq(self, value: Any) -> Dict[str, Any]:
            return {"attribute": self.name, "op": "eq", "value": value}


class _MissingBoto3Object:
    """Stub to provide clearer errors when boto3 clients/resources are missing."""

    def __init__(self, kind: str, service_name: str):
        self._kind = kind
        self._service = service_name
        exceptions_cls = type("exceptions", (), {})
        if service_name == "ssm":
            class ParameterNotFound(Exception):
                """Raised when an SSM parameter is missing."""

                pass

            exceptions_cls.ParameterNotFound = ParameterNotFound
        self.exceptions = exceptions_cls

    def __getattr__(self, item: str) -> Any:
        raise RuntimeError(
            f"boto3 is required to use AWS {self._kind} '{self._service}'. "
            "Install boto3 in the runtime or provide a mock during testing."
        )


def _create_boto3_client(service_name: str) -> Any:
    if boto3 is None:
        return _MissingBoto3Object("client", service_name)
    try:
        return boto3.client(service_name)
    except Exception:
        return _MissingBoto3Object("client", service_name)


def _create_boto3_resource(service_name: str) -> Any:
    if boto3 is None:
        return _MissingBoto3Object("resource", service_name)
    try:
        return boto3.resource(service_name)
    except Exception:
        return _MissingBoto3Object("resource", service_name)

VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
if VENDOR_DIR not in sys.path and os.path.isdir(VENDOR_DIR):
    sys.path.append(VENDOR_DIR)

import requests  # noqa: E402  pylint: disable=wrong-import-position

LOGGER_LEVELS = {
    "info": 20,
    "warning": 30,
    "critical": 50,
}

REQUEST_TIMEOUT = (10, 30)
SSM_CACHE: Dict[str, Tuple[str, float]] = {}
SSM_CLIENT = _create_boto3_client("ssm")
DDB_RESOURCE = _create_boto3_resource("dynamodb")
REQUEST_SESSION = requests.Session()
TIMEZONE = ZoneInfo("Australia/Melbourne")
VALID_MODES = {"create", "update", "auto"}
SSM_CACHE_TTL_SECONDS = int(os.environ.get("SSM_CACHE_TTL_SECONDS", "300"))
MAX_PLAYLIST_PAGES = int(os.environ.get("MAX_PLAYLIST_PAGES", "20"))
PROMPT_VERSION = "rotation-v1"
BATCH_PROMPT_VERSION = "batch-v1"
OPENAI_TRANSPORT: Optional[Callable[[Dict[str, Any]], requests.Response]] = None
HTTP_BACKOFF_KEY = "__http_backoff__"
PENDING_PLAYLIST_PREFIX = "to_be_created#"

RUN_ID_VAR: ContextVar[str] = ContextVar("run_id", default="")
JOB_ID_VAR: ContextVar[str] = ContextVar("job_id", default="")
CONTEXT_VAR: ContextVar[Any] = ContextVar("lambda_context", default=None)

ENV_CONFIG = {
    "spotify_client_id_param": os.environ.get("PARAM_SPOTIFY_CLIENT_ID"),
    "spotify_refresh_token_param": os.environ.get("PARAM_SPOTIFY_REFRESH_TOKEN"),
    "openai_api_key_param": os.environ.get("PARAM_OPENAI_API_KEY"),
    "request_api_key_param": os.environ.get("PARAM_REQUEST_API_KEY"),
    "default_market": os.environ.get("DEFAULT_MARKET", "AU"),
    "default_playlist_public": os.environ.get(
        "DEFAULT_PLAYLIST_PUBLIC", "false"
    ).lower()
    == "true",
    "playlist_state_table": os.environ.get("PLAYLIST_STATE_TABLE", "playlistbot_state"),
    "openai_throttle_ms": int(os.environ.get("OPENAI_THROTTLE_MS", "2500")),
    # Retry guards ensure we do not sleep longer than the remaining Lambda budget.
    "retry_safety_margin_ms": int(os.environ.get("RETRY_SAFETY_MARGIN_MS", "5000")),
    "min_request_budget_ms": int(os.environ.get("MIN_REQUEST_BUDGET_MS", "8000")),
    "handler_timeout_seconds": int(os.environ.get("HANDLER_TIMEOUT_SECONDS", "600")),
    "scheduled_openai_max_attempts": int(os.environ.get("SCHEDULED_OPENAI_MAX_ATTEMPTS", "1")),
}


def _clean_string(value: Optional[str]) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def build_playlist_display_name(template_name: str, subgenre: Optional[str]) -> str:
    base = _clean_string(template_name)
    if not base:
        raise ValueError("template_name is required")
    sub = _clean_string(subgenre)
    return f"{base} - {sub}" if sub else base


def log(level: str, msg: str, **details: Any) -> None:
    prefix = {
        "info": "[info]",
        "warning": "[warning]",
        "critical": "[critical]",
    }.get(level, "[info]")
    ctx = {}
    run_id = RUN_ID_VAR.get()
    job_id = JOB_ID_VAR.get()
    if run_id:
        ctx["run_id"] = run_id
    if job_id:
        ctx["job_id"] = job_id
    payload = {**ctx, **details} if details or ctx else None
    suffix = f" {json.dumps(payload, sort_keys=True, default=str)}" if payload else ""
    print(f"{prefix} {msg}{suffix}")


def get_remaining_time_ms() -> Optional[int]:
    ctx = CONTEXT_VAR.get()
    if not ctx:
        return None
    getter = getattr(ctx, "get_remaining_time_in_millis", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None


def log_runtime_identity(context: Any) -> None:
    log(
        "info",
        "lambda_identity",
        function_name=os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
        function_version=os.getenv("AWS_LAMBDA_FUNCTION_VERSION"),
        invoked_function_arn=getattr(context, "invoked_function_arn", None),
        aws_request_id=getattr(context, "aws_request_id", None),
    )


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    run_id = uuid.uuid4().hex
    token = RUN_ID_VAR.set(run_id)
    ctx_token = CONTEXT_VAR.set(context)
    invocation_type = (
        "scheduled"
        if is_scheduled_event(event)
        else "http"
        if is_apigw_http_event(event)
        else "unknown"
    )
    log(
        "info",
        "invocation_start",
        invocation_type=invocation_type,
        aws_request_id=getattr(context, "aws_request_id", None),
        function_name=getattr(context, "function_name", None),
        function_version=getattr(context, "function_version", None),
        timeout_seconds=ENV_CONFIG.get("handler_timeout_seconds"),
    )
    log_runtime_identity(context)
    try:
        if invocation_type == "scheduled":
            log("info", "Scheduled invocation detected")
            result = process_scheduled_event()
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result),
            }
        if invocation_type == "http":
            response = process_event(event)
        else:
            summary_keys = list(event.keys())[:10]
            log(
                "warning",
                "Unknown event type received",
                keys=summary_keys,
            )
            return build_response(
                400, {"status": "error", "message": "unknown event source"}
            )
    except HTTPError as exc:
        log(
            "warning", "Handled HTTP error", status=exc.status_code, message=exc.message
        )
        return build_response(exc.status_code, exc.to_body(), exc.headers)
    except Exception as exc:  # pylint: disable=broad-except
        log("critical", "Unhandled exception", error=str(exc))
        return build_response(
            502, {"status": "error", "message": "internal server error"}
        )
    finally:
        RUN_ID_VAR.reset(token)
        CONTEXT_VAR.reset(ctx_token)

    return response


def process_event(event: Dict[str, Any]) -> Dict[str, Any]:
    http_context = event.get("requestContext", {}).get("http")

    if http_context:
        method = http_context.get("method")
        path = http_context.get("path")
    else:
        method = event.get("httpMethod")
        path = event.get("resource") or event.get("path")

    log("info", "Incoming request", path=path)

    normalized_path = (path or "").rstrip("/") or "/"
    if method != "POST" or not normalized_path.endswith("/playlist"):
        raise HTTPError(404, "route not found")

    headers = {
        k.lower(): v
        for k, v in (event.get("headers") or {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }

    api_key = headers.get("x-api-key")
    if not api_key:
        raise HTTPError(401, "missing x-api-key header")

    expected_key = get_secure_parameter("request_api_key_param")
    if api_key != expected_key:
        raise HTTPError(401, "invalid api key")

    content_type = headers.get("content-type", "").split(";")[0].strip()
    if content_type != "application/json":
        raise HTTPError(400, "content-type must be application/json")

    table = DDB_RESOURCE.Table(ENV_CONFIG["playlist_state_table"])
    now_epoch = int(time.time())
    backoff_epoch = fetch_next_eligible_epoch(table, HTTP_BACKOFF_KEY)
    if backoff_epoch and backoff_epoch > now_epoch:
        seconds_remaining = max(backoff_epoch - now_epoch, 1)
        log(
            "info",
            "http_rate_limit_backoff",
            seconds_remaining=seconds_remaining,
            next_eligible_at_epoch=backoff_epoch,
        )
        body = {
            "error": "openai_rate_limited",
            "reason": "rate_limit_backoff",
            "next_eligible_at_epoch": backoff_epoch,
            "retry_after_s": seconds_remaining,
            "run_id": RUN_ID_VAR.get(),
        }
        return build_response(
            429,
            body,
            headers={"Retry-After": str(seconds_remaining)},
        )

    payload = parse_json_body(event)

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not (1 <= len(prompt) <= 500):
        raise HTTPError(400, "prompt must be a string between 1 and 500 characters")


    mode_raw = payload.get("mode", "auto")
    if mode_raw is None:
        mode_raw = "auto"
    if not isinstance(mode_raw, str):
        raise HTTPError(400, "mode must be one of create, update, or auto")
    mode = mode_raw.lower()
    if mode not in VALID_MODES:
        raise HTTPError(400, "mode must be one of create, update, or auto")

    playlist_name_raw = payload.get("playlist_name")
    if playlist_name_raw is not None and not isinstance(playlist_name_raw, str):
        raise HTTPError(400, "playlist_name must be a string when provided")

    playlist_name = (
        playlist_name_raw.strip() if isinstance(playlist_name_raw, str) else None
    )
    if playlist_name == "":
        playlist_name = None
    playlist_id_raw = payload.get("playlist_id")
    if playlist_id_raw is not None and not isinstance(playlist_id_raw, str):
        raise HTTPError(400, "playlist_id must be a string when provided")
    playlist_id = (
        playlist_id_raw.strip() if isinstance(playlist_id_raw, str) else None
    )
    if playlist_id == "":
        playlist_id = None

    if not playlist_id:
        should_attempt_update = mode == "update" or (mode == "auto" and playlist_name)
        if mode == "update" and not playlist_name:
            raise HTTPError(400, "playlist_name is required when mode is update")
    else:
        should_attempt_update = True

    body = generate_playlist_response(
        prompt=prompt,
        playlist_name=playlist_name,
        playlist_id=playlist_id,
        should_attempt_update=should_attempt_update,
    )

    return build_response(200, body)


def parse_json_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body = event.get("body")
    if body is None:
        raise HTTPError(400, "request body is required")

    if event.get("isBase64Encoded"):
        try:
            body = base64.b64decode(body).decode("utf-8")
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPError(400, "invalid base64 body") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPError(400, "body must be valid JSON") from exc

    if not isinstance(data, dict):
        raise HTTPError(400, "body must be a JSON object")

    return data


def request_playlist_spec(
    prompt: str,
    transport: Optional[Callable[[Dict[str, Any]], requests.Response]] = None,
    allow_rate_limit_retries: bool = True,
) -> Dict[str, Any]:
    api_key = get_secure_parameter("openai_api_key_param")
    prompt_hash = sha256_hash(prompt)

    def log_call(attempt: int) -> None:
        log(
            "info",
            "openai_api_call",
            call_type="single",
            prompt_hash=prompt_hash,
            prompt_length_chars=len(prompt),
            attempt=attempt,
        )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "temperature": 0.55,
        "messages": [
            {
                "role": "system",
                "content": (
                    """You are a music director that produces structured Spotify playlists.
                    Respond ONLY with strict JSON using the schema: {\\"base_name\\", \\"description\\", \\"tracks\\"}. 
                    Each track must include artist and title fields. Generate 30-60 original tracks, avoid duplicates, remixes unless asked, 
                    skip tribute/karaoke/cover versions, and keep a coherent energy arc for the provided prompt."""
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    def _make_request() -> requests.Response:
        if transport:
            return transport(payload)
        if OPENAI_TRANSPORT:
            return OPENAI_TRANSPORT(payload)
        return REQUEST_SESSION.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )

    resp = openai_request_with_retries(
        _make_request,
        remaining_time_ms_fn=get_remaining_time_ms,
        allow_rate_limit_retries=allow_rate_limit_retries,
        log_call=log_call,
    )

    if resp.status_code >= 400:
        log("warning", "OpenAI request failed", status=resp.status_code, body=resp.text)
        raise HTTPError(502, "failed to generate playlist spec")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        spec = json.loads(content)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log("critical", "Unexpected OpenAI response", response=resp.text)
        raise HTTPError(502, "invalid response from OpenAI") from exc

    log(
        "info",
        "openai_playlist_spec",
        call_type="single",
        prompt_hash=prompt_hash,
        track_count=len(spec.get("tracks") or []),
        spec=spec,
    )

    validate_spec(spec)
    return spec


def request_batch_playlist_specs(
    batch_jobs: List[Dict[str, Any]],
    *,
    transport: Optional[Callable[[Dict[str, Any]], requests.Response]] = None,
    max_attempts: Optional[int] = None,
) -> Dict[str, Dict[str, Any]]:
    if not batch_jobs:
        return {}

    api_key = get_secure_parameter("openai_api_key_param")
    payload_body = {"jobs": batch_jobs}
    jobs_text = json.dumps(payload_body, ensure_ascii=False)
    prompt_hash = sha256_hash(jobs_text)
    log(
        "info",
        "openai_batch_request_metadata",
        model="gpt-4o-mini",
        prompt_version=BATCH_PROMPT_VERSION,
        prompt_hash=prompt_hash,
        prompt_length_chars=len(jobs_text),
        playlist_count=len(batch_jobs),
        playlist_ids=[job.get("playlist_id") for job in batch_jobs],
    )
    def log_call(attempt: int) -> None:
        log(
            "info",
            "openai_api_call",
            call_type="batch",
            prompt_hash=prompt_hash,
            prompt_length_chars=len(jobs_text),
            playlist_count=len(batch_jobs),
            attempt=attempt,
        )
    system_message = (
        "You are a music director that returns ONLY strict JSON. "
        'The JSON schema is {"results":[{"playlist_id","base_name","description","tracks"}]}. '
        "Each track must contain artist and title. Avoid duplicates, covers, karaoke, or tribute versions."
    )
    user_message = (
        "Process every playlist job in the following JSON payload. "
        "For each job, follow the provided prompt text and generate between 30-60 tracks. "
        "Return JSON ONLY with a results array containing one entry per playlist_id.\n"
        f"{jobs_text}"
    )
    payload = {
        "model": "gpt-4o-mini",
        "response_format": {"type": "json_object"},
        "temperature": 0.55,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }

    def _make_request() -> requests.Response:
        if transport:
            return transport(payload)
        if OPENAI_TRANSPORT:
            return OPENAI_TRANSPORT(payload)
        return REQUEST_SESSION.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=REQUEST_TIMEOUT,
        )

    attempts = (
        max_attempts
        if isinstance(max_attempts, int) and max_attempts > 0
        else ENV_CONFIG.get("scheduled_openai_max_attempts", 1)
    )
    resp = openai_request_with_retries(
        _make_request,
        max_attempts=attempts,
        remaining_time_ms_fn=get_remaining_time_ms,
        allow_rate_limit_retries=False,
        log_call=log_call,
    )

    if resp.status_code >= 400:
        log("warning", "OpenAI batch request failed", status=resp.status_code, body=resp.text)
        raise HTTPError(502, "failed to generate playlist batch spec")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        log("critical", "Unexpected OpenAI batch response", response=resp.text)
        raise HTTPError(502, "invalid response from OpenAI batch") from exc

    results = parsed.get("results")
    if not isinstance(results, list):
        raise HTTPError(502, "invalid response from OpenAI batch")

    mapped: Dict[str, Dict[str, Any]] = {}
    for item in results:
        playlist_id = item.get("playlist_id")
        if isinstance(playlist_id, str):
            mapped[playlist_id] = item
            log(
                "info",
                "openai_playlist_spec",
                call_type="batch",
                playlist_id=playlist_id,
                track_count=len(item.get("tracks") or []),
                spec=item,
            )

    return mapped


def validate_spec(spec: Dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise HTTPError(502, "OpenAI did not return an object")

    if not isinstance(spec.get("base_name"), str) or not spec["base_name"].strip():
        raise HTTPError(502, "OpenAI base_name missing")

    if not isinstance(spec.get("description"), str):
        raise HTTPError(502, "OpenAI description missing")

    tracks = spec.get("tracks")
    if not isinstance(tracks, list) or not tracks:
        raise HTTPError(502, "OpenAI tracks missing")

    for track in tracks:
        if not isinstance(track, dict):
            raise HTTPError(502, "OpenAI track malformed")
        if not isinstance(track.get("artist"), str) or not isinstance(
            track.get("title"), str
        ):
            raise HTTPError(502, "OpenAI track missing fields")


def refresh_spotify_access_token() -> str:
    client_id = get_secure_parameter("spotify_client_id_param")
    refresh_param_name = _get_parameter_name("spotify_refresh_token_param")
    refresh_token = ssm_get_parameter(refresh_param_name)

    token, rotated_token, error_code = request_spotify_token(client_id, refresh_token)
    if token:
        if rotated_token:
            persist_refresh_token(refresh_param_name, rotated_token)
        return token

    if error_code == "invalid_grant":
        log("warning", "Refresh token invalid, forcing SSM re-read")
        fresh_refresh_token = ssm_get_parameter(refresh_param_name, force_refresh=True)
        token, rotated_token, error_code = request_spotify_token(
            client_id, fresh_refresh_token
        )
        if token:
            if rotated_token:
                persist_refresh_token(refresh_param_name, rotated_token)
            return token
        if error_code == "invalid_grant":
            raise HTTPError(502, "spotify refresh token invalid after retry")

    raise HTTPError(502, "unable to refresh Spotify token")


def request_spotify_token(
    client_id: str, refresh_token: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    data = {
        "grant_type": "refresh_token",
        "client_id": client_id,
        "refresh_token": refresh_token,
    }

    resp = REQUEST_SESSION.post(
        "https://accounts.spotify.com/api/token",
        data=data,
        timeout=REQUEST_TIMEOUT,
    )

    payload: Dict[str, Any] = {}
    try:
        payload = resp.json()
    except ValueError:
        payload = {}

    if resp.status_code >= 400:
        error_code, description = parse_spotify_error(payload)
        log(
            "warning",
            "Spotify token refresh failed",
            status=resp.status_code,
            error=error_code,
            description=description,
        )
        if resp.status_code == 400 and error_code == "invalid_grant":
            return None, None, "invalid_grant"
        raise HTTPError(502, "unable to refresh Spotify token")

    token = payload.get("access_token")
    if not token:
        log("critical", "Spotify response missing access_token")
        raise HTTPError(502, "Spotify token missing")

    rotated_token = payload.get("refresh_token")
    return token, rotated_token, None


def fetch_spotify_user(access_token: str) -> Dict[str, Any]:
    resp = spotify_get("https://api.spotify.com/v1/me", access_token)
    return resp


def fetch_playlist_by_id(access_token: str, playlist_id: str) -> Dict[str, Any]:
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    resp = REQUEST_SESSION.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"fields": "id,name,external_urls"},
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code == 404:
        raise HTTPError(404, "playlist_id not found")
    if resp.status_code >= 400:
        log(
            "warning",
            "Spotify GET playlist by id failed",
            status=resp.status_code,
            body=resp.text,
            playlist_id=playlist_id,
        )
        raise HTTPError(502, "spotify request failed")
    payload = resp.json()
    if not payload.get("id"):
        raise HTTPError(502, "spotify playlist payload missing id")
    return payload


def find_playlist_by_name(
    access_token: str, desired_name: Optional[str]
) -> Optional[Dict[str, Any]]:
    if not desired_name:
        return None

    normalized_target = _normalize_playlist_name(desired_name)
    if not normalized_target:
        return None

    url = "https://api.spotify.com/v1/me/playlists"
    params = {"limit": 50, "offset": 0}
    page_count = 0
    exact_matches: List[Dict[str, Any]] = []
    contains_matches: List[Dict[str, Any]] = []

    while page_count < MAX_PLAYLIST_PAGES:
        payload = spotify_get(url, access_token, params=params)
        for item in payload.get("items", []):
            candidate_name = item.get("name") or ""
            normalized_candidate = _normalize_playlist_name(candidate_name)
            if not normalized_candidate:
                continue

            if normalized_candidate == normalized_target:
                exact_matches.append(item)
            elif normalized_target in normalized_candidate:
                contains_matches.append(item)

        if exact_matches:
            break

        next_url = payload.get("next")
        page_count += 1
        if next_url:
            params["offset"] += params["limit"]
        else:
            break

    if page_count >= MAX_PLAYLIST_PAGES and not exact_matches and not contains_matches:
        log(
            "warning",
            "Reached max playlist pages without finding match",
            desired=desired_name,
        )

    if exact_matches:
        if len(exact_matches) > 1:
            log(
                "warning",
                "Multiple playlists matched exact name",
                desired=desired_name,
                count=len(exact_matches),
            )
        return exact_matches[0]

    if len(contains_matches) == 1:
        return contains_matches[0]

    if len(contains_matches) > 1:
        log(
            "warning",
            "Multiple playlists matched partial name",
            desired=desired_name,
            count=len(contains_matches),
        )

    return None


def create_playlist(
    access_token: str, user_id: str, name: str, description: Optional[str]
) -> Dict[str, Any]:
    payload = {
        "name": name,
        "description": description or "",
        "public": ENV_CONFIG["default_playlist_public"],
    }
    url = f"https://api.spotify.com/v1/users/{user_id}/playlists"
    resp = spotify_post(url, access_token, json_body=payload)
    return resp


def update_playlist_metadata(
    access_token: str,
    playlist_id: str,
    name: str,
    description: Optional[str],
) -> None:
    payload: Dict[str, Any] = {
        "name": name,
        "public": ENV_CONFIG["default_playlist_public"],
    }
    if isinstance(description, str):
        payload["description"] = description
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
    spotify_put(url, access_token, json_body=payload)


def resolve_track_uris(
    access_token: str, tracks: List[Dict[str, str]], market: str
) -> Tuple[List[str], List[str]]:
    uris: List[str] = []
    unmatched: List[str] = []

    for track in tracks:
        query = f"track:{track['title']} artist:{track['artist']}"
        search_params = {
            "q": query,
            "type": "track",
            "limit": 10,
            "market": market,
        }
        payload = spotify_get(
            "https://api.spotify.com/v1/search", access_token, params=search_params
        )
        best_uri = score_best_track(payload.get("tracks", {}).get("items", []), track)
        if best_uri:
            uris.append(best_uri)
        else:
            unmatched.append(f"{track['artist']} – {track['title']}")

    log(
        "info",
        "spotify_track_resolution",
        requested=len(tracks or []),
        matched=len(uris),
        unmatched=len(unmatched),
        unmatched_tracks=unmatched,
    )

    return uris, unmatched


def score_best_track(
    candidates: List[Dict[str, Any]], desired: Dict[str, str]
) -> Optional[str]:
    target_artist = desired["artist"].lower()
    target_title = desired["title"].lower()
    best_score = -1.0
    best_uri: Optional[str] = None

    for item in candidates:
        name = item.get("name", "").lower()
        artists = ", ".join(a.get("name", "") for a in item.get("artists", [])).lower()
        score = 0.0
        if target_title in name:
            score += 2.0
        if name == target_title:
            score += 2.0
        if target_artist in artists:
            score += 2.0
        if any(
            target_artist == a.get("name", "").lower() for a in item.get("artists", [])
        ):
            score += 1.0
        popularity = item.get("popularity") or 0
        score += popularity / 100.0
        if score > best_score:
            best_score = score
            best_uri = item.get("uri")

    return best_uri


def replace_playlist_contents(
    access_token: str, playlist_id: str, uris: List[str]
) -> None:
    chunks = [uris[i : i + 100] for i in range(0, len(uris), 100)] or [[]]
    first_chunk = chunks[0]
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    spotify_put(url, access_token, json_body={"uris": first_chunk})
    for chunk in chunks[1:]:
        spotify_post(url, access_token, json_body={"uris": chunk})


def add_tracks_to_playlist(
    access_token: str, playlist_id: str, uris: List[str]
) -> None:
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks"
    for i in range(0, len(uris), 100):
        chunk = uris[i : i + 100]
        spotify_post(url, access_token, json_body={"uris": chunk})


def spotify_get(
    url: str, access_token: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    resp = REQUEST_SESSION.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        log(
            "warning",
            "Spotify GET failed",
            url=url,
            status=resp.status_code,
            body=resp.text,
        )
        raise HTTPError(502, "spotify request failed")
    return resp.json()


def spotify_post(
    url: str, access_token: str, json_body: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    resp = REQUEST_SESSION.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(json_body or {}),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        log(
            "warning",
            "Spotify POST failed",
            url=url,
            status=resp.status_code,
            body=resp.text,
        )
        raise HTTPError(502, "spotify request failed")
    return resp.json() if resp.text else {}


def spotify_put(
    url: str, access_token: str, json_body: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    resp = REQUEST_SESSION.put(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(json_body or {}),
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 400:
        log(
            "warning",
            "Spotify PUT failed",
            url=url,
            status=resp.status_code,
            body=resp.text,
        )
        raise HTTPError(502, "spotify request failed")
    return resp.json() if resp.text else {}


def get_secure_parameter(config_key: str, force_refresh: bool = False) -> str:
    param_name = _get_parameter_name(config_key)
    return ssm_get_parameter(param_name, force_refresh=force_refresh)


def build_response(
    status_code: int, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    final_headers = {"Content-Type": "application/json"}
    if headers:
        final_headers.update(headers)
    return {
        "statusCode": status_code,
        "headers": final_headers,
        "body": json.dumps(body),
    }


def is_scheduled_event(event: Dict[str, Any]) -> bool:
    # Example EventBridge events:
    # {"source":"aws.events","detail-type":"Scheduled Event","detail":{...}}
    # {"source":"aws.scheduler","detail-type":"Scheduled Event","detail":{...}}
    if not isinstance(event, dict):
        return False
    if event.get("scheduled"):
        return True
    source = event.get("source")
    detail_type = event.get("detail-type")
    if source in {"aws.events", "aws.scheduler"} and detail_type == "Scheduled Event":
        return True
    return False


def is_apigw_http_event(event: Dict[str, Any]) -> bool:
    if not isinstance(event, dict):
        return False
    request_context = event.get("requestContext")
    if isinstance(request_context, dict):
        if "http" in request_context or "stage" in request_context:
            return True
    if "httpMethod" in event or "resource" in event or "path" in event:
        return True
    return False


def _safe_get(container: Dict[str, Any], *keys: str) -> Optional[Any]:
    value: Any = container
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


class HTTPError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        details: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        raw_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details
        self.headers = headers or {}
        self.raw_body = raw_body

    def to_body(self) -> Dict[str, Any]:
        if self.raw_body is not None:
            return self.raw_body
        payload = {"status": "error", "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return payload


def persist_refresh_token(param_name: str, value: str) -> None:
    now = time.time()
    SSM_CACHE[param_name] = (value, now + SSM_CACHE_TTL_SECONDS)
    try:
        ssm_put_parameter(param_name, value, secure=True)
    except Exception as exc:  # pylint: disable=broad-except
        log(
            "warning",
            "Failed to persist rotated Spotify refresh token",
            error=str(exc),
        )


def parse_spotify_error(
    payload: Optional[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(payload, dict):
        return None, None
    error_value = payload.get("error")
    error_description = payload.get("error_description")
    if isinstance(error_value, dict):
        error_value = error_value.get("message")
    if isinstance(error_description, dict):
        error_description = error_description.get("message")
    if isinstance(error_value, str):
        error_value = error_value.strip()
    if isinstance(error_description, str):
        error_description = error_description.strip()
    return error_value, error_description


def ssm_get_parameter(name: str, force_refresh: bool = False) -> str:
    now = time.time()
    if not force_refresh:
        cached = SSM_CACHE.get(name)
        if cached and cached[1] > now:
            return cached[0]

    try:
        response = SSM_CLIENT.get_parameter(Name=name, WithDecryption=True)
    except SSM_CLIENT.exceptions.ParameterNotFound as exc:  # type: ignore[attr-defined]
        raise HTTPError(502, f"ssm parameter {name} not found") from exc

    value = response["Parameter"]["Value"]
    SSM_CACHE[name] = (value, now + SSM_CACHE_TTL_SECONDS)
    return value


def ssm_put_parameter(name: str, value: str, secure: bool = True) -> None:
    params = {
        "Name": name,
        "Value": value,
        "Overwrite": True,
        "Type": "SecureString" if secure else "String",
    }
    SSM_CLIENT.put_parameter(**params)


def _get_parameter_name(config_key: str) -> str:
    param_name = ENV_CONFIG.get(config_key)
    if not param_name:
        raise HTTPError(502, f"missing environment configuration for {config_key}")
    return param_name


def _normalize_playlist_name(value: Optional[str]) -> str:
    if not value:
        return ""
    text = value.strip()
    quotes = "\"'“”‘’"
    while text and text[0] in quotes:
        text = text[1:]
    while text and text[-1:] and text[-1] in quotes:
        text = text[:-1]
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)
    return text.casefold()


def build_history_sets(
    history_entries: List[Dict[str, Any]], window_days: int
) -> Tuple[set, set, set]:
    relevant = (
        history_entries[-window_days:]
        if window_days > 0 and len(history_entries) > window_days
        else history_entries
    )
    track_keys: set = set()
    artist_keys: set = set()
    uris: set = set()
    for entry in relevant:
        for key in entry.get("track_keys", []):
            if isinstance(key, str):
                track_keys.add(key)
        for key in entry.get("artist_keys", []):
            if isinstance(key, str):
                artist_keys.add(key)
        for uri in entry.get("uris", []):
            if isinstance(uri, str):
                uris.add(uri)
    return track_keys, artist_keys, uris


def filter_tracks_with_constraints(
    tracks: List[Dict[str, str]],
    exclude_track_keys: set,
    exclude_artist_keys: set,
    max_tracks_per_artist: int,
    track_count: int,
) -> List[Dict[str, str]]:
    final_tracks: List[Dict[str, str]] = []
    track_block = set(exclude_track_keys)
    artist_block = set(exclude_artist_keys)
    artist_counts: Dict[str, int] = defaultdict(int)
    exclusion_log: Dict[str, List[str]] = defaultdict(list)

    def record_exclusion(reason: str, track: Dict[str, str]) -> None:
        artist = (track.get("artist") or "").strip() or "unknown artist"
        title = (track.get("title") or "").strip() or "unknown title"
        label = f"{artist} – {title}"
        if len(exclusion_log[reason]) < 50:
            exclusion_log[reason].append(label)

    for track in tracks or []:
        artist = track.get("artist")
        title = track.get("title")
        if not artist or not title:
            record_exclusion("missing_fields", track)
            continue
        track_key = _track_key(artist, title)
        artist_key = _artist_key(artist)
        if track_key in track_block:
            record_exclusion("duplicate_track", track)
            continue
        if artist_key in artist_block:
            record_exclusion("artist_blocked", track)
            continue
        if artist_counts[artist_key] >= max_tracks_per_artist:
            record_exclusion("artist_quota_reached", track)
            continue

        final_tracks.append({"artist": artist.strip(), "title": title.strip()})
        track_block.add(track_key)
        artist_block.add(artist_key)
        artist_counts[artist_key] += 1

        if len(final_tracks) >= track_count:
            break

    if not final_tracks:
        raise HTTPError(
            502,
            "no tracks available after applying exclusions",
            {"reason": "no_tracks_available_after_exclusions"},
        )

    if len(final_tracks) < track_count:
        log(
            "warning",
            "Scheduled generation produced fewer tracks than requested",
            requested=track_count,
            produced=len(final_tracks),
        )
    exclusion_summary = [
        {
            "reason": reason,
            "count": len(entries),
            "samples": entries[:10],
        }
        for reason, entries in exclusion_log.items()
    ]
    log(
        "info",
        "track_filtering_summary",
        considered=len(tracks or []),
        requested=track_count,
        produced=len(final_tracks),
        exclusion_summary=exclusion_summary,
    )

    return final_tracks


def compute_next_eligible_epoch(retry_after_s: int) -> int:
    delay = max(int(retry_after_s or 60), 30)
    jitter = random.randint(0, 15)
    return int(time.time()) + delay + jitter


def set_rate_limit_backoff(
    table: Any, playlist_id: str, retry_after_s: int
) -> int:
    next_epoch = compute_next_eligible_epoch(retry_after_s)
    table.update_item(
        Key={"playlist_id": playlist_id},
        UpdateExpression="SET next_eligible_at_epoch = :ts",
        ExpressionAttributeValues={":ts": Decimal(str(next_epoch))},
    )
    log(
        "info",
        "rate_limit_backoff_set",
        playlist_id=playlist_id,
        next_eligible_at_epoch=next_epoch,
        retry_after_s=retry_after_s,
    )
    return next_epoch


def fetch_next_eligible_epoch(table: Any, playlist_id: str) -> Optional[int]:
    try:
        response = table.get_item(Key={"playlist_id": playlist_id})
    except Exception as exc:  # pylint: disable=broad-except
        log(
            "warning",
            "Failed to fetch backoff state",
            playlist_id=playlist_id,
            error=str(exc),
        )
        return None
    item = response.get("Item")
    if not item:
        return None
    return _to_int(item.get("next_eligible_at_epoch"), 0)


def ensure_playlist_id(
    table: Any,
    item: Dict[str, Any],
    playlist_id: Optional[str],
    template_name: str,
    theme_name: Optional[str],
) -> str:
    pending = False
    if playlist_id and not playlist_id.startswith(PENDING_PLAYLIST_PREFIX):
        return playlist_id
    if playlist_id and playlist_id.startswith(PENDING_PLAYLIST_PREFIX):
        pending = True

    access_token = refresh_spotify_access_token()
    user = fetch_spotify_user(access_token)
    base_name = build_playlist_display_name(template_name, theme_name)
    description = item.get("base_prompt")
    playlist = create_playlist(
        access_token, user["id"], base_name, description if isinstance(description, str) else None
    )
    new_playlist_id = playlist["id"]
    item["playlist_id"] = new_playlist_id
    try:
        table.put_item(Item=item)
    except Exception as exc:  # pylint: disable=broad-except
        log(
            "warning",
            "Failed to persist new playlist mapping",
            playlist_id=new_playlist_id,
            error=str(exc),
        )
    placeholder_id = playlist_id if pending else None
    if placeholder_id and placeholder_id != new_playlist_id:
        try:
            table.delete_item(Key={"playlist_id": placeholder_id})
        except Exception as exc:  # pylint: disable=broad-except
            log(
                "warning",
                "Failed to delete placeholder playlist entry",
                playlist_id=placeholder_id,
                error=str(exc),
            )
    log(
        "info",
        "bootstrap_playlist_created",
        playlist_id=new_playlist_id,
        playlist_name=playlist.get("name"),
    )
    return new_playlist_id


def advance_rotation_cursor(
    table: Any,
    playlist_id: str,
    current_cursor: int,
    rotation_length: int,
) -> None:
    if rotation_length <= 0:
        return
    next_cursor = (current_cursor + 1) % rotation_length
    try:
        table.update_item(
            Key={"playlist_id": playlist_id},
            UpdateExpression="SET rotation_cursor = :cursor",
            ExpressionAttributeValues={":cursor": next_cursor},
        )
    except Exception as exc:  # pylint: disable=broad-except
        log(
            "warning",
            "Failed to advance rotation cursor",
            playlist_id=playlist_id,
            error=str(exc),
        )
def build_scheduled_prompt(
    base_prompt: str,
    needed: int,
    exclude_track_keys: List[str],
    exclude_artist_keys: List[str],
    theme_prompt: Optional[str] = None,
    extra_instructions: Optional[List[str]] = None,
) -> str:
    parts = [
        base_prompt.strip(),
        f"Generate at least {needed} unique tracks (artist and title). Provide only original studio versions and keep the existing vibe.",
        "Return the full list; do not truncate or omit entries.",
    ]
    if theme_prompt:
        parts.append(theme_prompt.strip())
    if exclude_track_keys:
        limited_tracks = exclude_track_keys[:200]
        parts.append(
            "Do NOT include any of the following tracks:\n- "
            + "\n- ".join(limited_tracks)
        )
    if exclude_artist_keys:
        limited_artists = exclude_artist_keys[:200]
        parts.append(
            "Avoid these artists where possible:\n- " + "\n- ".join(limited_artists)
        )
    if extra_instructions:
        parts.extend(extra_instructions)
    return "\n\n".join(parts)


def update_playlist_history(
    table: Any,
    playlist_id: str,
    history_entries: List[Dict[str, Any]],
    new_entry: Dict[str, Any],
    window_days: int,
) -> None:
    entries = (history_entries or []) + [new_entry]
    if window_days > 0 and len(entries) > window_days:
        entries = entries[-window_days:]

    table.update_item(
        Key={"playlist_id": playlist_id},
        UpdateExpression="SET history_entries = :entries",
        ExpressionAttributeValues={":entries": entries},
    )


def _track_key(artist: str, title: str) -> str:
    return f"{artist.strip().casefold()}|{title.strip().casefold()}"


def _artist_key(artist: str) -> str:
    return artist.strip().casefold()


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_novelty_metrics(
    uris: List[str],
    artist_keys: List[str],
    yesterday_uris: set,
    window_uris: set,
    window_artist_keys: set,
    track_count_expected: int,
) -> Dict[str, float]:
    denom = track_count_expected or len(uris) or 1
    uri_set = set(uris)
    overlap_yesterday = len(uri_set & yesterday_uris) / denom
    overlap_window = len(uri_set & window_uris) / denom
    artist_set = set(artist_keys)
    new_artists = len(artist_set - window_artist_keys)
    total_artists = len(artist_set) or 1
    new_artist_ratio = new_artists / total_artists
    return {
        "overlap_yesterday": round(overlap_yesterday, 4),
        "overlap_window": round(overlap_window, 4),
        "new_artist_ratio": round(new_artist_ratio, 4),
    }


def openai_request_with_retries(
    request_fn: Callable[[], requests.Response],
    max_attempts: int = 6,
    max_sleep_s: float = 30.0,
    remaining_time_ms_fn: Optional[Callable[[], Optional[int]]] = None,
    allow_rate_limit_retries: bool = True,
    log_call: Optional[Callable[[int], None]] = None,
) -> requests.Response:
    last_error: Optional[Exception] = None
    safety_margin_ms = ENV_CONFIG.get("retry_safety_margin_ms", 5000)
    min_request_budget_ms = ENV_CONFIG.get("min_request_budget_ms", 8000)

    def _remaining_ms() -> Optional[int]:
        if not remaining_time_ms_fn:
            return None
        try:
            value = remaining_time_ms_fn()
        except Exception:
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _abort_due_to_budget(phase: str, attempt: int, sleep_s: Optional[float] = None):
        remaining_ms = _remaining_ms()
        if remaining_ms is None:
            return
        log_kwargs = {
            "phase": phase,
            "attempt": attempt,
            "remaining_ms": remaining_ms,
        }
        if sleep_s is not None:
            log_kwargs["sleep_s"] = round(sleep_s, 2)
            log_kwargs["safety_margin_ms"] = safety_margin_ms
        log(
            "warning",
            "Not enough time left for OpenAI retry budget; aborting",
            **log_kwargs,
        )
        raise HTTPError(
            502,
            "openai_rate_limited_timeout_budget",
            {
                "reason": "openai_rate_limited_timeout_budget",
                "phase": phase,
                "attempt": attempt,
                "remaining_ms": remaining_ms,
            },
        )

    def _ensure_request_budget(attempt: int) -> None:
        remaining_ms = _remaining_ms()
        if remaining_ms is None:
            return
        if remaining_ms < min_request_budget_ms:
            _abort_due_to_budget("pre_request", attempt)

    def _ensure_sleep_budget(attempt: int, sleep_s: float) -> None:
        if sleep_s <= 0:
            return
        remaining_ms = _remaining_ms()
        if remaining_ms is None:
            return
        total_sleep_ms = int(sleep_s * 1000)
        if total_sleep_ms + safety_margin_ms > remaining_ms:
            _abort_due_to_budget("sleep", attempt, sleep_s=sleep_s)

    for attempt in range(1, max_attempts + 1):
        _ensure_request_budget(attempt)
        if log_call:
            try:
                log_call(attempt)
            except Exception:
                pass
        try:
            resp = request_fn()
            log(
                "info",
                "OpenAI request completed",
                attempt=attempt,
                status=resp.status_code,
            )
        except requests.RequestException as exc:
            last_error = exc
            if attempt == max_attempts:
                raise HTTPError(
                    502,
                    "openai_unavailable",
                    {"reason": "network_error", "error": str(exc)},
                ) from exc
            sleep_s = compute_retry_sleep(attempt, max_sleep_s)
            _ensure_sleep_budget(attempt, sleep_s)
            log(
                "warning",
                "OpenAI request failed - retrying",
                attempt=attempt,
                sleep_s=round(sleep_s, 2),
                error=str(exc),
            )
            sleep_for(sleep_s)
            continue

        if resp.status_code == 429:
            retry_after_header = resp.headers.get("Retry-After")
            retry_after_value: Optional[float] = None
            if retry_after_header:
                try:
                    retry_after_value = float(retry_after_header)
                except ValueError:
                    retry_after_value = None
            if not allow_rate_limit_retries or attempt == max_attempts:
                raise HTTPError(
                    502,
                    "openai_rate_limited",
                    {
                        "reason": "openai_rate_limited",
                        "status": 429,
                        "retry_after_s": (
                            int(retry_after_value)
                            if retry_after_value is not None
                            else None
                        ),
                    },
                )
            sleep_s = compute_retry_sleep(attempt, max_sleep_s, retry_after_header)
            _ensure_sleep_budget(attempt, sleep_s)
            log(
                "warning",
                "OpenAI rate limited - retrying",
                attempt=attempt,
                sleep_s=round(sleep_s, 2),
            )
            sleep_for(sleep_s)
            continue

        if resp.status_code >= 500:
            if attempt == max_attempts:
                raise HTTPError(
                    502,
                    "openai_unavailable",
                    {"reason": "openai_unavailable", "status": resp.status_code},
                )
            sleep_s = compute_retry_sleep(attempt, max_sleep_s)
            _ensure_sleep_budget(attempt, sleep_s)
            log(
                "warning",
                "OpenAI 5xx - retrying",
                attempt=attempt,
                sleep_s=round(sleep_s, 2),
                status=resp.status_code,
            )
            sleep_for(sleep_s)
            continue

        return resp

    raise HTTPError(
        502,
        "openai_unavailable",
        {"reason": "openai_unknown_error", "error": str(last_error)},
    )


def compute_retry_sleep(
    attempt: int, max_sleep_s: float, retry_after: Optional[str] = None
) -> float:
    if retry_after:
        try:
            base = min(float(retry_after), max_sleep_s)
        except ValueError:
            base = min(max_sleep_s, 2 * (2 ** (attempt - 1)))
    else:
        base = min(max_sleep_s, 2 * (2 ** (attempt - 1)))
    jitter = random.uniform(0, max(0.5 * base, 0.1))
    return min(base + jitter, max_sleep_s)


def sleep_for(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def sha256_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
def process_scheduled_event() -> Dict[str, Any]:
    table_name = ENV_CONFIG.get("playlist_state_table")
    table = DDB_RESOURCE.Table(table_name)
    items: List[Dict[str, Any]] = []
    last_key = None

    while True:
        scan_kwargs: Dict[str, Any] = {"FilterExpression": Attr("enabled").eq(True)}
        if last_key:
            scan_kwargs["ExclusiveStartKey"] = last_key
        response = table.scan(**scan_kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break

    configured_playlist_count = len(items)
    log(
        "info",
        "scheduled_config_loaded",
        config_source="dynamo",
        identifier=table_name,
        configured_playlist_count=configured_playlist_count,
    )

    processed = 0
    succeeded = 0
    failed = 0
    deferred = 0
    skipped = 0
    failures: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    throttle_ms = ENV_CONFIG.get("openai_throttle_ms", 0)
    job_contexts: List[Dict[str, Any]] = []
    batch_requests: List[Dict[str, Any]] = []

    for item in items:
        processed += 1
        playlist_id = item.get("playlist_id")
        base_prompt = item.get("base_prompt")
        template_name = _clean_string(item.get("config_name"))
        job_id = f"{playlist_id}:{RUN_ID_VAR.get()}" if playlist_id else RUN_ID_VAR.get()
        history_entries = item.get("history_entries") or []
        genre = item.get("genre") or item.get("base_genre") or "unspecified"
        previous_subgenre = history_entries[-1].get("selected_subgenre") if history_entries else None
        rotation_themes = item.get("rotation_themes") or []
        missing_fields = []
        if not playlist_id:
            missing_fields.append("playlist_id")
        if not base_prompt:
            missing_fields.append("base_prompt")
        if not template_name:
            missing_fields.append("config_name")
        if missing_fields:
            failed += 1
            log(
                "warning",
                "Scheduled item missing required fields",
                item=item,
            )
            log(
                "info",
                "scheduled_job_skipped",
                playlist_id=playlist_id,
                job_id=job_id,
                reason="invalid_config",
            )
            failures.append(
                {
                    "playlist_id": playlist_id or "unknown",
                    "reason": "invalid_config",
                    "status": 400,
                }
            )
            results.append(
                {
                    "playlist_id": playlist_id or "unknown",
                    "job_id": job_id,
                    "genre": genre,
                    "selected_subgenre": None,
                    "prompt_hash": None,
                    "outcome": "failed",
                    "error_category": "invalid_config",
                    "error_message": f"missing {', '.join(missing_fields)}",
                }
            )
            continue
        rotation_length = len(rotation_themes)
        rotation_cursor = _to_int(item.get("rotation_cursor"), 0)
        rotation_index = None
        theme_prompt = None
        theme_name = None
        if rotation_length:
            rotation_index = rotation_cursor % rotation_length
            theme = rotation_themes[rotation_index]
            if isinstance(theme, dict):
                theme_name = theme.get("name")
                theme_prompt = theme.get("prompt")
        playlist_display_name = build_playlist_display_name(template_name, theme_name)

        playlist_id = ensure_playlist_id(
            table, item, playlist_id, template_name, theme_name
        )
        job_id = f"{playlist_id}:{RUN_ID_VAR.get()}"

        next_eligible_epoch = _to_int(item.get("next_eligible_at_epoch"), 0)
        now_epoch = int(time.time())
        if next_eligible_epoch and next_eligible_epoch > now_epoch:
            skipped += 1
            job_token = JOB_ID_VAR.set(job_id or "")
            log(
                "info",
                "scheduled_job_skipped",
                playlist_id=playlist_id,
                job_id=job_id,
                reason="rate_limit_backoff",
                next_eligible_at_epoch=next_eligible_epoch,
            )
            JOB_ID_VAR.reset(job_token)
            results.append(
                {
                    "playlist_id": playlist_id,
                    "job_id": job_id,
                    "genre": genre,
                    "selected_subgenre": None,
                    "prompt_hash": None,
                    "outcome": "skipped",
                    "error_category": "rate_limit_backoff",
                    "next_eligible_at_epoch": next_eligible_epoch,
                }
            )
            continue

        window_days = _to_int(item.get("window_days"), 14)
        track_count = _to_int(item.get("track_count"), 50)
        max_tracks_per_artist = _to_int(item.get("max_tracks_per_artist"), 1)
        max_overlap_yesterday = _to_float(item.get("max_overlap_yesterday"), 0.35)
        max_overlap_window = _to_float(item.get("max_overlap_window"), 0.55)
        min_new_artists_window = _to_float(item.get("min_new_artists_window"), 0.60)
        exclude_track_keys, exclude_artist_keys, window_uris = build_history_sets(
            history_entries, window_days
        )
        window_artist_keys = set(exclude_artist_keys)
        yesterday_entry = history_entries[-1] if history_entries else None
        yesterday_uris = set(yesterday_entry.get("uris", [])) if yesterday_entry else set()

        job_context = {
            "playlist_id": playlist_id,
            "job_id": job_id,
            "playlist_name": playlist_display_name,
            "genre": genre,
            "template_name": template_name,
            "base_prompt": base_prompt,
            "history_entries": history_entries,
            "window_days": window_days,
            "track_count": track_count,
            "max_tracks_per_artist": max_tracks_per_artist,
            "max_overlap_yesterday": max_overlap_yesterday,
            "max_overlap_window": max_overlap_window,
            "min_new_artists_window": min_new_artists_window,
            "exclude_track_keys": exclude_track_keys,
            "exclude_artist_keys": exclude_artist_keys,
            "window_uris": window_uris,
            "window_artist_keys": window_artist_keys,
            "yesterday_uris": yesterday_uris,
            "theme_name": theme_name,
            "theme_prompt": theme_prompt,
            "rotation_index": rotation_index,
            "rotation_cursor": rotation_cursor,
            "rotation_length": rotation_length,
            "previous_subgenre": previous_subgenre,
            "rotation_themes": rotation_themes,
        }
        job_contexts.append(job_context)

        job_token = JOB_ID_VAR.set(job_id or "")
        log(
            "info",
            "scheduled_job_context",
            playlist_id=playlist_id,
            playlist_name=playlist_display_name,
            genre=genre,
            previous_subgenre=previous_subgenre,
            selected_subgenre=theme_name,
            rotation_index=rotation_index,
        )
        JOB_ID_VAR.reset(job_token)

        prompt_needed = min(
            track_count * 2,
            track_count + max(15, track_count // 2),
        )
        job_context["prompt_track_target"] = prompt_needed

        extra_instructions = [
            (
                "Respect novelty guardrails: "
                f"overlap_yesterday <= {max_overlap_yesterday:.2f}, "
                f"overlap_window <= {max_overlap_window:.2f}, "
                f"new_artist_ratio >= {min_new_artists_window:.2f}, "
                f"max {max_tracks_per_artist} tracks per artist."
            )
        ]
        prompt_text = build_scheduled_prompt(
            base_prompt,
            prompt_needed,
            list(exclude_track_keys),
            list(exclude_artist_keys),
            theme_prompt=theme_prompt,
            extra_instructions=extra_instructions,
        )
        prompt_hash = sha256_hash(prompt_text)
        job_context["prompt_hash"] = prompt_hash
        job_context["prompt_text"] = prompt_text
        job_context["extra_instructions"] = extra_instructions

        job_token = JOB_ID_VAR.set(job_id or "")
        log(
            "info",
            "openai_prompt_metadata",
            model="gpt-4o-mini",
            prompt_version=PROMPT_VERSION,
            prompt_hash=prompt_hash,
            prompt_length_chars=len(prompt_text),
            genre=genre,
            selected_subgenre=theme_name,
        )
        JOB_ID_VAR.reset(job_token)

        batch_requests.append(
            {
                "playlist_id": playlist_id,
                "prompt": prompt_text,
                "track_count": track_count,
                "genre": genre,
                "selected_subgenre": theme_name,
            }
        )

    batch_results: Dict[str, Dict[str, Any]] = {}
    batch_error: Optional[HTTPError] = None
    openai_called = False
    openai_attempts = 0
    if batch_requests:
        try:
            openai_called = True
            openai_attempts = 1
            batch_results = request_batch_playlist_specs(batch_requests)
        except HTTPError as exc:
            batch_error = exc
            log(
                "warning",
                "OpenAI batch request failed",
                status=exc.status_code,
                message=exc.message,
            )

    for idx, job in enumerate(job_contexts, start=1):
        playlist_id = job["playlist_id"]
        job_id = job["job_id"]
        job_token = JOB_ID_VAR.set(job_id or "")
        job_start = time.perf_counter()
        log(
            "info",
            "scheduled_job_start",
            playlist_id=playlist_id,
            job_id=job_id,
        )
        try:
            if batch_error:
                raise HTTPError(
                    batch_error.status_code,
                    batch_error.message,
                    batch_error.details,
                )

            spec_entry = batch_results.get(playlist_id)
            if not spec_entry:
                raise HTTPError(
                    502,
                    "openai_bad_batch_response",
                    {"reason": "openai_bad_batch_response"},
                )

            validate_spec(spec_entry)
            job_result = run_scheduled_playlist(job, spec_entry, table)
            succeeded += 1
            duration_ms = int((time.perf_counter() - job_start) * 1000)
            advance_rotation_cursor(
                table,
                playlist_id,
                job.get("rotation_cursor", 0),
                job.get("rotation_length", 0),
            )
            log(
                "info",
                "scheduled_job_complete",
                playlist_id=playlist_id,
                job_id=job_id,
                outcome="succeeded",
                duration_ms=duration_ms,
            )
            results.append(
                {
                    "playlist_id": playlist_id,
                    "job_id": job_id,
                    "genre": job_result.get("genre"),
                    "selected_subgenre": job_result.get("selected_subgenre"),
                    "prompt_hash": job_result.get("prompt_hash"),
                    "outcome": "succeeded",
                }
            )
        except HTTPError as exc:
            log(
                "warning",
                "Scheduled playlist regeneration failed",
                playlist_id=playlist_id,
                message=exc.message,
                status=exc.status_code,
            )
            reason = (exc.details or {}).get("reason")
            if not reason and exc.message == "no tracks available after applying exclusions":
                reason = "no_tracks_available_after_exclusions"
            if reason == "openai_rate_limited":
                deferred += 1
                retry_after_s = _to_int((exc.details or {}).get("retry_after_s"), 60)
                next_epoch = set_rate_limit_backoff(table, playlist_id, retry_after_s)
                results.append(
                    {
                        "playlist_id": playlist_id,
                        "job_id": job_id,
                        "genre": job.get("genre"),
                        "selected_subgenre": job.get("theme_name"),
                        "prompt_hash": job.get("prompt_hash"),
                        "outcome": "deferred",
                        "error_category": reason,
                        "retry_after_s": retry_after_s,
                        "next_eligible_at_epoch": next_epoch,
                    }
                )
                duration_ms = int((time.perf_counter() - job_start) * 1000)
                log(
                    "warning",
                    "scheduled_job_complete",
                    playlist_id=playlist_id,
                    job_id=job_id,
                    outcome="deferred",
                    duration_ms=duration_ms,
                    error_category=reason,
                )
                continue
            if reason == "no_tracks_available_after_exclusions":
                try:
                    retry_spec = prepare_spec_for_next_theme(job)
                except HTTPError as retry_exc:
                    log(
                        "warning",
                        "Retry preparation failed",
                        playlist_id=playlist_id,
                        message=retry_exc.message,
                    )
                    retry_spec = None
                if retry_spec:
                    try:
                        retry_start = time.perf_counter()
                        job_result = run_scheduled_playlist(job, retry_spec, table)
                        succeeded += 1
                        advance_rotation_cursor(
                            table,
                            playlist_id,
                            job.get("rotation_cursor", 0),
                            job.get("rotation_length", 0),
                        )
                        duration_ms = int((time.perf_counter() - retry_start) * 1000)
                        log(
                            "info",
                            "scheduled_job_complete",
                            playlist_id=playlist_id,
                            job_id=job_id,
                            outcome="succeeded",
                            duration_ms=duration_ms,
                            retry_mode="next_theme",
                        )
                        results.append(
                            {
                                "playlist_id": playlist_id,
                                "job_id": job_id,
                                "genre": job_result.get("genre"),
                                "selected_subgenre": job_result.get("selected_subgenre"),
                                "prompt_hash": job_result.get("prompt_hash"),
                                "outcome": "succeeded",
                                "retry_mode": "next_theme",
                            }
                        )
                        continue
                    except HTTPError as retry_exc:
                        log(
                            "warning",
                            "Retry with next theme failed",
                            playlist_id=playlist_id,
                            message=retry_exc.message,
                            status=retry_exc.status_code,
                        )
                        exc = retry_exc
                        reason = (retry_exc.details or {}).get("reason") or reason
            else:
                status_code = getattr(exc, "status_code", 500)
                error_category = reason or exc.message
                failures.append(
                    {
                        "playlist_id": playlist_id,
                        "reason": error_category,
                        "status": status_code,
                    }
                )
                results.append(
                    {
                        "playlist_id": playlist_id,
                        "job_id": job_id,
                        "genre": job.get("genre"),
                        "selected_subgenre": job.get("theme_name"),
                        "prompt_hash": job.get("prompt_hash"),
                        "outcome": "failed",
                        "error_category": error_category,
                        "error_message": str(exc)[:200],
                        "status_code": status_code,
                    }
                )
                duration_ms = int((time.perf_counter() - job_start) * 1000)
                log(
                    "warning",
                    "scheduled_job_complete",
                    playlist_id=playlist_id,
                    job_id=job_id,
                    outcome="failed",
                    duration_ms=duration_ms,
                    error_category=error_category,
                    status_code=status_code,
                )
                continue
            failed += 1
        except Exception as exc:  # pylint: disable=broad-except
            failed += 1
            log(
                "warning",
                "Unexpected error during scheduled regeneration",
                playlist_id=playlist_id,
                error=str(exc),
            )
            failures.append(
                {
                    "playlist_id": playlist_id,
                    "reason": "exception",
                    "error": str(exc)[:120],
                    "status": 500,
                }
            )
            results.append(
                {
                    "playlist_id": playlist_id,
                    "job_id": job_id,
                    "genre": job.get("genre"),
                    "selected_subgenre": job.get("theme_name"),
                    "prompt_hash": job.get("prompt_hash"),
                    "outcome": "failed",
                    "error_category": "exception",
                    "error_message": str(exc)[:200],
                }
            )
            duration_ms = int((time.perf_counter() - job_start) * 1000)
            log(
                "warning",
                "scheduled_job_complete",
                playlist_id=playlist_id,
                job_id=job_id,
                outcome="failed",
                duration_ms=duration_ms,
                error_category="exception",
            )
        finally:
            JOB_ID_VAR.reset(job_token)
            if throttle_ms and idx < len(job_contexts):
                sleep_for(throttle_ms / 1000.0)

    log(
        "info",
        "scheduled_openai_summary",
        openai_called=openai_called,
        openai_attempts=openai_attempts,
        succeeded=succeeded,
        failed=failed,
        deferred=deferred,
        skipped=skipped,
    )

    summary = {
        "run_id": RUN_ID_VAR.get(),
        "status": "scheduled-complete",
        "processed": processed,
        "succeeded": succeeded,
        "failed": failed,
        "deferred": deferred,
        "skipped": skipped,
        "failures": failures,
        "results": results,
    }
    log("info", "Scheduled regeneration summary", **summary)
    return summary


def generate_playlist_response(
    prompt: str,
    playlist_name: Optional[str],
    playlist_id: Optional[str],
    should_attempt_update: bool,
) -> Dict[str, Any]:
    try:
        spec = request_playlist_spec(
            prompt, allow_rate_limit_retries=False
        )
    except HTTPError as exc:
        reason = (exc.details or {}).get("reason") if exc.details else None
        if reason == "openai_rate_limited":
            retry_after_s = _to_int((exc.details or {}).get("retry_after_s"), 60)
            table = DDB_RESOURCE.Table(ENV_CONFIG["playlist_state_table"])
            next_epoch = set_rate_limit_backoff(
                table, HTTP_BACKOFF_KEY, retry_after_s
            )
            seconds_remaining = max(next_epoch - int(time.time()), 1)
            body = {
                "error": "openai_rate_limited",
                "reason": "rate_limit_backoff",
                "next_eligible_at_epoch": next_epoch,
                "retry_after_s": seconds_remaining,
                "run_id": RUN_ID_VAR.get(),
            }
            raise HTTPError(
                429,
                "openai_rate_limited",
                details={"reason": "openai_rate_limited"},
                headers={"Retry-After": str(seconds_remaining)},
                raw_body=body,
            ) from exc
        raise
    suffix = datetime.now(tz=TIMEZONE).strftime(" %d-%m-%Y")
    desired_name = f"{spec['base_name']}{suffix}"

    access_token = refresh_spotify_access_token()
    user = fetch_spotify_user(access_token)

    playlist_data: Optional[Dict[str, Any]] = None
    effective_mode = "create"

    if playlist_id:
        playlist_data = fetch_playlist_by_id(access_token, playlist_id)
        effective_mode = "update"
    elif should_attempt_update and playlist_name:
        candidate = find_playlist_by_name(access_token, playlist_name)
        if candidate:
            playlist_data = candidate
            effective_mode = "update"
        else:
            log(
                "info",
                "Playlist requested for update not found; creating new playlist",
                requested_name=playlist_name,
            )

    if playlist_data is None:
        playlist_data = create_playlist(
            access_token, user["id"], desired_name, spec.get("description")
        )
        effective_mode = "create"

    final_playlist_name = playlist_data["name"]
    playlist_id = playlist_data["id"]
    playlist_url = playlist_data.get("external_urls", {}).get("spotify")

    uris, unmatched = resolve_track_uris(
        access_token, spec["tracks"], ENV_CONFIG["default_market"]
    )

    if effective_mode == "update":
        replace_playlist_contents(access_token, playlist_id, uris)
    else:
        add_tracks_to_playlist(access_token, playlist_id, uris)

    result = {
        "status": "done",
        "message": f'done, the playlist name is "{final_playlist_name}"',
        "playlist_name": final_playlist_name,
        "playlist_id": playlist_id,
        "playlist_url": playlist_url,
        "matched": len(uris),
        "unmatched": unmatched,
        "mode_effective": effective_mode,
    }
    log(
        "info",
        "Playlist updated via API",
        mode=effective_mode,
        playlist_id=playlist_id,
        matches=len(uris),
        unmatched=len(unmatched),
    )
    return result


def run_scheduled_playlist(
    job: Dict[str, Any],
    spec: Dict[str, Any],
    table: Any,
) -> Dict[str, Any]:
    playlist_id = job["playlist_id"]
    playlist_name = job.get("playlist_name")
    genre = job.get("genre")
    track_count = job.get("track_count", 50)
    max_tracks_per_artist = job.get("max_tracks_per_artist", 1)
    max_overlap_yesterday = job.get("max_overlap_yesterday", 0.35)
    max_overlap_window = job.get("max_overlap_window", 0.55)
    min_new_artists_window = job.get("min_new_artists_window", 0.60)
    history_entries = job.get("history_entries") or []
    window_days = job.get("window_days", 14)
    exclude_track_keys = job.get("exclude_track_keys", set())
    exclude_artist_keys = job.get("exclude_artist_keys", set())
    window_uris = job.get("window_uris", set())
    window_artist_keys = job.get("window_artist_keys", set())
    yesterday_uris = job.get("yesterday_uris", set())
    theme_name = job.get("theme_name")
    prompt_hash = job.get("prompt_hash")
    template_name = job.get("template_name") or ""
    base_prompt = job.get("base_prompt")

    filtered_tracks = filter_tracks_with_constraints(
        spec.get("tracks"),
        exclude_track_keys,
        exclude_artist_keys,
        max_tracks_per_artist,
        track_count,
    )

    access_token = refresh_spotify_access_token()
    uris, unmatched = resolve_track_uris(
        access_token, filtered_tracks, ENV_CONFIG["default_market"]
    )
    artist_keys = [_artist_key(t["artist"]) for t in filtered_tracks]
    metrics = compute_novelty_metrics(
        uris=uris,
        artist_keys=artist_keys,
        yesterday_uris=yesterday_uris,
        window_uris=window_uris,
        window_artist_keys=window_artist_keys,
        track_count_expected=track_count,
    )

    playlist_data = fetch_playlist_by_id(access_token, playlist_id)
    desired_playlist_name: Optional[str] = None
    try:
        desired_playlist_name = build_playlist_display_name(template_name, theme_name)
    except ValueError:
        desired_playlist_name = playlist_name
    if desired_playlist_name and playlist_data.get("name") != desired_playlist_name:
        description_text = spec.get("description") or (
            base_prompt if isinstance(base_prompt, str) else None
        )
        try:
            update_playlist_metadata(
                access_token,
                playlist_id,
                desired_playlist_name,
                description_text,
            )
            playlist_data["name"] = desired_playlist_name
            playlist_name = desired_playlist_name
        except Exception as exc:  # pylint: disable=broad-except
            log(
                "warning",
                "Failed to update playlist metadata",
                playlist_id=playlist_id,
                error=str(exc),
            )
    else:
        playlist_name = playlist_data.get("name")
    playlist_url = playlist_data.get("external_urls", {}).get("spotify")

    replace_playlist_contents(access_token, playlist_id, uris)

    date_str = datetime.now(tz=TIMEZONE).strftime("%Y-%m-%d")
    track_keys = [_track_key(t["artist"], t["title"]) for t in filtered_tracks]

    update_playlist_history(
        table,
        playlist_id,
        history_entries,
        {
            "date": date_str,
            "track_keys": track_keys,
            "artist_keys": artist_keys,
            "uris": uris,
            "selected_subgenre": theme_name,
        },
        window_days,
    )

    final_pass = (
        metrics["overlap_yesterday"] <= max_overlap_yesterday
        and metrics["overlap_window"] <= max_overlap_window
        and metrics["new_artist_ratio"] >= min_new_artists_window
    )
    if not final_pass:
        log(
            "warning",
            "Scheduled playlist metrics exceeded thresholds",
            playlist_id=playlist_id,
            metrics=metrics,
        )

    log(
        "info",
        "Scheduled playlist updated",
        playlist_id=playlist_id,
        playlist_name=playlist_name,
        theme=theme_name,
        metrics=metrics,
        tracks=len(filtered_tracks),
        matched=len(uris),
        unmatched=len(unmatched),
        playlist_url=playlist_url,
    )

    return {
        "prompt_hash": prompt_hash,
        "selected_subgenre": theme_name,
        "genre": genre,
    }


def prepare_spec_for_next_theme(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    rotation_length = job.get("rotation_length", 0)
    themes = job.get("rotation_themes") or []
    if rotation_length <= 1 or not themes:
        return None
    next_index = (job.get("rotation_index") or 0) + 1
    next_index %= rotation_length
    theme = themes[next_index] if next_index < len(themes) else None
    if not isinstance(theme, dict):
        return None
    theme_name = theme.get("name")
    theme_prompt = theme.get("prompt")
    prompt_target = job.get("prompt_track_target") or job.get("track_count") or 50
    prompt_text = build_scheduled_prompt(
        job.get("base_prompt") or "",
        prompt_target,
        list(job.get("exclude_track_keys", [])),
        list(job.get("exclude_artist_keys", [])),
        theme_prompt=theme_prompt,
        extra_instructions=job.get("extra_instructions"),
    )
    prompt_hash = sha256_hash(prompt_text)
    job["theme_name"] = theme_name
    job["prompt_text"] = prompt_text
    job["prompt_hash"] = prompt_hash
    job["rotation_index"] = next_index
    job["rotation_cursor"] = next_index
    job["playlist_name"] = build_playlist_display_name(
        job.get("template_name") or "", theme_name
    )
    log(
        "info",
        "scheduled_retry_next_theme",
        playlist_id=job.get("playlist_id"),
        job_id=job.get("job_id"),
        next_theme=theme_name,
        prompt_hash=prompt_hash,
    )
    spec = request_playlist_spec(prompt_text, allow_rate_limit_retries=False)
    return spec
