import base64
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import boto3
from zoneinfo import ZoneInfo

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
SSM_CLIENT = boto3.client("ssm")
REQUEST_SESSION = requests.Session()
TIMEZONE = ZoneInfo("Australia/Melbourne")
VALID_MODES = {"create", "update", "auto"}
SSM_CACHE_TTL_SECONDS = int(os.environ.get("SSM_CACHE_TTL_SECONDS", "300"))
MAX_PLAYLIST_PAGES = int(os.environ.get("MAX_PLAYLIST_PAGES", "20"))

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
}


def log(level: str, msg: str, **details: Any) -> None:
    prefix = {
        "info": "[info]",
        "warning": "[warning]",
        "critical": "[critical]",
    }.get(level, "[info]")
    suffix = f" {json.dumps(details, sort_keys=True, default=str)}" if details else ""
    print(f"{prefix} {msg}{suffix}")


def handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    try:
        response = process_event(event)
    except HTTPError as exc:
        log(
            "warning", "Handled HTTP error", status=exc.status_code, message=exc.message
        )
        return build_response(exc.status_code, exc.to_body())
    except Exception as exc:  # pylint: disable=broad-except
        log("critical", "Unhandled exception", error=str(exc))
        return build_response(
            502, {"status": "error", "message": "internal server error"}
        )

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

    if playlist_id:
        should_attempt_update = True
    else:
        should_attempt_update = mode == "update" or (mode == "auto" and playlist_name)
        if mode == "update" and not playlist_name:
            raise HTTPError(400, "playlist_name is required when mode is update")

    spec = request_playlist_spec(prompt)
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

    body = {
        "status": "done",
        "message": f'done, the playlist name is "{final_playlist_name}"',
        "playlist_name": final_playlist_name,
        "playlist_id": playlist_id,
        "playlist_url": playlist_url,
        "matched": len(uris),
        "unmatched": unmatched,
        "mode_effective": effective_mode,
    }

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


def request_playlist_spec(prompt: str) -> Dict[str, Any]:
    api_key = get_secure_parameter("openai_api_key_param")
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

    resp = REQUEST_SESSION.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=REQUEST_TIMEOUT,
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

    validate_spec(spec)
    return spec


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


def build_response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def _safe_get(container: Dict[str, Any], *keys: str) -> Optional[Any]:
    value: Any = container
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


class HTTPError(Exception):
    def __init__(
        self, status_code: int, message: str, details: Optional[Any] = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.details = details

    def to_body(self) -> Dict[str, Any]:
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

