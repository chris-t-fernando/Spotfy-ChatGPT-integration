# Prompt → Spotify Playlist Lambda

Python 3.11 AWS Lambda that turns a natural-language prompt into a curated Spotify playlist through OpenAI + Spotify APIs. Terraform in `../terraform` provisions the Lambda, IAM role, API Gateway REST API, and SSM parameters required for secrets.

## Architecture Overview
- **API Gateway REST API** exposes `POST /playlist` with an `x-api-key` header enforced at the edge and validated again in the Lambda.
- **Lambda** validates the payload, calls OpenAI for a playlist spec, refreshes a Spotify access token, finds/creates playlists, resolves tracks via Spotify Search, and writes playlist contents.
- **SSM Parameter Store** holds the shared secret, Spotify refresh credentials, and OpenAI API key (created as placeholder SecureStrings by Terraform).

## Spotify Refresh Token via PKCE
1. Register a Spotify app and add `http://localhost:8080/callback` to Redirect URIs.
2. Locally generate a `code_verifier` + `code_challenge` (e.g. with `python - <<'PY' ...`).
3. Open the authorize URL in your browser:
   ```
   https://accounts.spotify.com/authorize?client_id=<CLIENT_ID>&response_type=code&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback&scope=playlist-modify-private%20playlist-read-private&code_challenge=<CODE_CHALLENGE>&code_challenge_method=S256
   ```
4. Capture the `code` from the redirected URL and exchange it locally:
   ```bash
   curl -X POST https://accounts.spotify.com/api/token \
     -d grant_type=authorization_code \
     -d client_id=<CLIENT_ID> \
     -d code=<CODE_FROM_BROWSER> \
     -d redirect_uri=http://localhost:8080/callback \
     -d code_verifier=<CODE_VERIFIER>
   ```
5. Store the returned `refresh_token` securely; the Lambda only needs the refresh token + client_id.

## Secrets in SSM Parameter Store
Terraform creates placeholder SecureString parameters that **must** be overwritten out-of-band before invoking the API:
```
/playlistbot/spotify/client_id
/playlistbot/spotify/refresh_token
/playlistbot/openai/api_key
/playlistbot/security/api_key
```
Set the real values:
```bash
aws ssm put-parameter --name /playlistbot/spotify/client_id --type SecureString --value <client_id> --overwrite
aws ssm put-parameter --name /playlistbot/spotify/refresh_token --type SecureString --value <refresh_token> --overwrite
aws ssm put-parameter --name /playlistbot/openai/api_key --type SecureString --value <openai_key> --overwrite
aws ssm put-parameter --name /playlistbot/security/api_key --type SecureString --value <shared_secret> --overwrite
```
Do not manage `/playlistbot/spotify/refresh_token` via Terraform outputs or state—always rotate it with `aws ssm put-parameter` so concurrent Lambda refreshes can persist the latest token.

## Packaging Dependencies
The Lambda imports `requests`, so bundle dependencies before `terraform apply`:
```bash
rm -rf lambda/vendor
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r lambda/requirements.txt -t lambda/vendor
```
`terraform` zips the entire `lambda/` directory (including `vendor/`). Re-run the packaging step whenever dependencies change.

## Deploy with Terraform
```bash
cd terraform
terraform init
terraform plan -out tfplan
terraform apply tfplan
```
Outputs include the Lambda name, API invoke URL, and parameter names.

## Scheduled Playlist Regeneration
Terraform also provisions:
- **DynamoDB `playlistbot_state`** to store scheduled playlist configs + history
- **EventBridge rule `playlistbot-daily-regen`** (runs daily at 03:10 Australia/Melbourne, i.e., 17:10 UTC) that invokes the Lambda with `{"scheduled":true}`

### 1. Seed the DynamoDB table
Start with at least one playlist configuration. Example (Lo-fi Study with seven rotating sub-genres):
- `playlist_id` can be a placeholder formatted as `to_be_created#<slug>`. The Lambda will create the playlist on the first scheduled run, persist the real Spotify playlist id back into DynamoDB, and delete the placeholder entry.
- Include `rotation_cursor` (initially `"0"`) so the Lambda can remember which sub-genre runs next.
- Every template **must** include `config_name` (e.g., `"lofi"` or `"hiphop"`). During each scheduled run the Spotify playlist is renamed to `<config_name> - <current rotation theme>` (or just `<config_name>` when no rotation is defined), so pick a stable, human-friendly name.
Use DynamoDB attribute-value JSON (every attribute needs a type wrapper):
```json
{
  "playlist_id": {"S": "to_be_created#lofi_focus"},
  "enabled": {"BOOL": true},
  "config_name": {"S": "lofi"},
  "base_prompt": {"S": "Curate chilled lo-fi beats for deep focus with warm textures."},
  "rotation_themes": {
    "L": [
      {"M": {"name": {"S": "Ghibli Comfort"}, "prompt": {"S": "Blend Studio Ghibli inspired motifs."}}},
      {"M": {"name": {"S": "Neo Classical"}, "prompt": {"S": "Lean into soft piano & modern classical."}}},
      {"M": {"name": {"S": "Rainy Neon"}, "prompt": {"S": "Rain ambience with neon city vibes."}}},
      {"M": {"name": {"S": "Jazz Café"}, "prompt": {"S": "Dusty jazz-hop sampled warmth."}}},
      {"M": {"name": {"S": "Bossa Drift"}, "prompt": {"S": "Brazilian bossa undertones and nylon guitar."}}},
      {"M": {"name": {"S": "Ambient Lean"}, "prompt": {"S": "Soft pads + minimal percussion."}}},
      {"M": {"name": {"S": "Bit-crushed Nostalgia"}, "prompt": {"S": "8-bit textures and tape hiss."}}}
    ]
  },
  "window_days": {"N": "14"},
  "track_count": {"N": "50"},
  "max_tracks_per_artist": {"N": "1"},
  "max_overlap_yesterday": {"N": "0.35"},
  "max_overlap_window": {"N": "0.55"},
  "min_new_artists_window": {"N": "0.6"},
  "max_attempts": {"N": "4"},
  "history_entries": {"L": []},
  "rotation_cursor": {"N": "0"}
}
```
Insert via CLI:
```bash
aws dynamodb put-item \
  --table-name playlistbot_state \
  --item file://seed_lofi.json
```
Repeat for each playlist you want auto-rotated. Set `enabled=false` to pause.

### Field reference
- `window_days`: Size of the historical window used to compute novelty metrics and exclusion sets. Larger windows block reusing artists/tracks for longer, increasing freshness at the cost of OpenAI/Spotify matches.
- `track_count`: Target number of tracks to publish. We request more than this from OpenAI to account for filtering, but the playlist is trimmed to exactly `track_count` when possible.
- `max_tracks_per_artist`: Hard cap on how many tracks from the same artist can appear in a single scheduled run.
- `max_overlap_yesterday`: Max proportion (0–1) of today’s URIs that can overlap with the previous day’s URIs before the result is considered out-of-guardrail (logged as a warning).
- `max_overlap_window`: Same as above but measured against the last `window_days` entries.
- `min_new_artists_window`: Minimum fraction of artists that must be new compared to the rolling window. Raising this value forces more novel artists and can increase `artist_blocked` exclusions.
- `history_entries`: Rolling log of previous runs (auto-maintained). Do not edit manually unless seeding initial history.
- `rotation_cursor`: Pointer used to select the next entry in `rotation_themes`. The Lambda updates this automatically after each successful run.

### 2. How scheduling works
- EventBridge calls the Lambda once per day (`{"scheduled":true}`).
- Lambda scans `playlistbot_state` for `enabled=true` entries, determines the rotation theme for the day, and regenerates the Spotify playlist by `playlist_id`.
- Novelty guards enforce rolling no-repeat windows, maximum overlap, and minimum new artist ratios. Exclusions are passed into OpenAI prompts and enforced post-generation.
- OpenAI retries for scheduled runs are controlled via the Terraform variable `scheduled_openai_max_attempts` (default `1`). Bump this to `2`+ if the OpenAI API is intermittently unavailable and you want additional automatic retries.

### 3. Adding/changing sub-genres (rotation themes)
- Add additional objects in `rotation_themes`. The Lambda picks them deterministically using `(YYYYMMDD % len(rotation_themes))`, ensuring a predictable cycle.
- Update policy fields (`window_days`, `track_count`, etc.) to tune novelty pressure per playlist.
- `history_entries` are updated automatically after each run; do not edit manually unless seeding initial history.

## Calling the Endpoint
Replace `<API_URL>` and `<API_KEY>` with your deployment outputs. The request body supports:
- `prompt` (required): 1–500 character description of the vibe.
- `mode` (optional): `create`, `update`, or `auto` (default). Only `mode` + targeting fields drive behavior.
- `playlist_name` (optional): target playlist when `mode=update` (and no `playlist_id`) or when `mode=auto` and you want to update by name.
- `playlist_id` (optional): deterministic target. If provided, the Lambda updates that exact playlist ID and never creates a new playlist.

**Create new playlist explicitly**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "give me chill synthwave for late-night focus",
    "mode": "create"
  }'
```

**Update existing playlist by name (create-if-missing)**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "amp me up for a tempo run",
    "mode": "update",
    "playlist_name": "Tempo Run Booster"
  }'
```
If the playlist exists its contents are replaced without renaming. If it does not exist, a new playlist is created using the OpenAI-generated base name + date suffix.

**Auto mode (default)**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "give me acoustic background music for dinner",
    "playlist_name": "Sunday Roast"
  }'
```
Because no `mode` is provided, auto mode updates an existing playlist named “Sunday Roast” or creates a new one if it does not exist.

**Update the last playlist via playlist_id**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "refresh it with new releases",
    "mode": "update",
    "playlist_id": "37i9dQZF1DX4JAvHpjipBk"
  }'
```
If the playlist ID is invalid or not accessible, the Lambda returns HTTP 404 with `{"message":"playlist_id not found"}`.

Responses include:
```json
{
  "status": "done",
  "message": "done, the playlist name is \"Tempo Run Booster\"",
  "playlist_name": "Tempo Run Booster",
  "playlist_id": "37i9dQZF1DX4JAvHpjipBk",
  "playlist_url": "https://open.spotify.com/playlist/…",
  "matched": 48,
  "unmatched": ["Artist – Song"],
  "mode_effective": "update"
}
```

## Notes
- All outbound HTTP calls have 10s connect / 30s read timeouts.
- SSM values are cached between invocations for performance.
- Logs include `[info]`, `[warning]`, and `[critical]` prefixes for easier CloudWatch filtering.
