# Prompt → Spotify Playlist Lambda

Python 3.11 AWS Lambda that turns a natural-language prompt into a curated Spotify playlist via OpenAI + Spotify APIs. Terraform under `terraform/` provisions the Lambda, IAM role, API Gateway REST API, EventBridge schedule, and required SSM parameters.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Deployment Workflow](#deployment-workflow)
   - [Package Lambda Dependencies](#package-lambda-dependencies)
   - [Provision Infrastructure with Terraform](#provision-infrastructure-with-terraform)
3. [Configuration](#configuration)
   - [Secrets in SSM Parameter Store](#secrets-in-ssm-parameter-store)
   - [Spotify Refresh Token via PKCE](#spotify-refresh-token-via-pkce)
   - [Scheduled Playlist Configuration (DynamoDB)](#scheduled-playlist-configuration-dynamodb)
4. [Scheduled Regeneration](#scheduled-regeneration)
   - [How scheduling works](#how-scheduling-works)
   - [Managing rotation themes](#managing-rotation-themes)
   - [Manually triggering the scheduled Lambda](#manually-triggering-the-scheduled-lambda)
5. [Manual Playlist Generation API](#manual-playlist-generation-api)
6. [Operations, Debugging & Logging](#operations-debugging--logging)
7. [Reference Notes](#reference-notes)

## Architecture Overview
- **API Gateway REST API** exposes `POST /playlist` with an `x-api-key` header enforced at the edge and validated inside the Lambda.
- **Lambda** validates payloads, calls OpenAI for a playlist spec, refreshes a Spotify access token, ensures playlists exist, resolves tracks, and writes playlist contents. The same handler also processes scheduled events for recurring playlist refreshes.
- **EventBridge** invokes the Lambda daily (03:10 Australia/Melbourne / 17:10 UTC) with `{"scheduled": true}` to run every enabled playlist configuration.
- **DynamoDB (`playlistbot_state`)** stores playlist definitions, history entries, and rotation metadata.
- **SSM Parameter Store** holds secrets (API key, Spotify client credentials, OpenAI key) and exposes them to the Lambda via environment variables.

## Deployment Workflow
### Package Lambda Dependencies
Bundle third-party packages (currently `requests`) before applying Terraform, because the deployment zips the `lambda/` directory as-is.
```bash
rm -rf lambda/vendor
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r lambda/requirements.txt -t lambda/vendor
```
Re-run whenever dependencies change.

### Provision Infrastructure with Terraform
```bash
cd terraform
terraform init
terraform plan -out tfplan
terraform apply tfplan
```
Outputs include the Lambda name, invoke URL, and fully qualified SSM parameter names. Use the same workflow for redeployments after code changes (package → `terraform apply`).

## Configuration
### Secrets in SSM Parameter Store
Terraform seeds placeholder SecureStrings. Overwrite them with real values **before** invoking the Lambda:
```
/playlistbot/spotify/client_id
/playlistbot/spotify/refresh_token
/playlistbot/openai/api_key
/playlistbot/security/api_key
```
Example:
```bash
aws ssm put-parameter --name /playlistbot/spotify/client_id --type SecureString --value <client_id> --overwrite
aws ssm put-parameter --name /playlistbot/spotify/refresh_token --type SecureString --value <refresh_token> --overwrite
aws ssm put-parameter --name /playlistbot/openai/api_key --type SecureString --value <openai_key> --overwrite
aws ssm put-parameter --name /playlistbot/security/api_key --type SecureString --value <shared_secret> --overwrite
```
Do not surface the refresh token via Terraform outputs; Lambda rotates it automatically as Spotify issues new tokens.

### Spotify Refresh Token via PKCE
1. Register a Spotify app and add `http://localhost:8080/callback` to Redirect URIs.
2. Generate `code_verifier` + `code_challenge` locally (e.g., a short Python script).
3. Authorize in your browser:
   ```
   https://accounts.spotify.com/authorize?client_id=<CLIENT_ID>&response_type=code&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback&scope=playlist-modify-private%20playlist-read-private&code_challenge=<CODE_CHALLENGE>&code_challenge_method=S256
   ```
4. Exchange the returned `code`:
   ```bash
   curl -X POST https://accounts.spotify.com/api/token \
     -d grant_type=authorization_code \
     -d client_id=<CLIENT_ID> \
     -d code=<CODE_FROM_BROWSER> \
     -d redirect_uri=http://localhost:8080/callback \
     -d code_verifier=<CODE_VERIFIER>
   ```
5. Store the resulting `refresh_token` in SSM.

### Scheduled Playlist Configuration (DynamoDB)
Seed `playlistbot_state` with playlist definitions. Example entry (lo-fi rotation with placeholder playlist):
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
Load it via:
```bash
aws dynamodb put-item \
  --table-name playlistbot_state \
  --item file://seed_lofi.json
```
Notes:
- `playlist_id` may start as `to_be_created#slug`; Lambda replaces it with the real Spotify ID on the first run.
- `config_name` drives playlist naming (`<config_name> - <theme>` when rotations exist).
- Legacy novelty guard fields (`window_days`, overlap ratios, etc.) remain in the schema but are currently ignored—history is captured for observability only.

## Scheduled Regeneration
### How scheduling works
1. EventBridge invokes the Lambda with `{"scheduled": true}` once per day.
2. Lambda scans `playlistbot_state` for `enabled=true`, resolves the next rotation theme, generates a playlist spec via OpenAI, and overwrites the Spotify playlist contents.
3. `scheduled_openai_max_attempts` (Terraform variable, default `1`) controls how many times the handler will attempt new OpenAI specs per job when failures occur.
4. After a successful run, the Lambda advances `rotation_cursor` and logs novelty metrics.

### Managing rotation themes
- Append new objects to `rotation_themes` for each config to expand the deterministic cycle.
- `rotation_cursor` tracks the current index and is auto-updated.
- Adjust `track_count` or prompts at any time; changes take effect on the next run.

### Manually triggering the scheduled Lambda
Force a refresh outside the normal window (e.g., after seeding data):
```bash
LAMBDA_NAME=$(terraform -chdir=terraform output -raw playlistbot_lambda_name)
aws lambda invoke \
  --function-name "$LAMBDA_NAME" \
  --payload '{"scheduled": true}' \
  /tmp/scheduled-run.json
cat /tmp/scheduled-run.json
```
- The response mirrors the CloudWatch summary (`processed`, `succeeded`, etc.).
- Tail logs: `aws logs tail /aws/lambda/$LAMBDA_NAME --follow`.
- To refresh only certain playlists, temporarily update `enabled=false` for others and revert afterward.

## Manual Playlist Generation API
Invoke the REST API for ad-hoc playlists.

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
If the playlist exists its contents are replaced; otherwise a new playlist is created using the OpenAI base name + date suffix.

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
`mode` omitted → auto. Updates an existing playlist named “Sunday Roast” or creates it if missing.

**Target by `playlist_id`**
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
Invalid IDs return HTTP 404 with `{"message":"playlist_id not found"}`.

Example success payload:
```json
{
  "status": "done",
  "message": "done, the playlist name is \"Tempo Run Booster\"",
  "playlist_name": "Tempo Run Booster",
  "playlist_id": "37i9dQZF1DX4JAvHpjipBk",
  "playlist_url": "https://open.spotify.com/playlist/...",
  "matched": 48,
  "unmatched": ["Artist – Song"],
  "mode_effective": "update"
}
```

## Operations, Debugging & Logging
- HTTP clients use 10 s connect / 30 s read timeouts. OpenAI calls automatically retry with exponential backoff while respecting remaining Lambda time.
- SSM lookups are cached for 5 minutes per parameter to reduce latency.
- CloudWatch log entries are prefixed with `[info]`, `[warning]`, or `[critical]` for easy filtering.
- Scheduled runs summarize outcomes in JSON (`status`, `processed`, `succeeded`, `failed`, `deferred`, `skipped`). The same payload is returned when you manually invoke the scheduled event.
- To troubleshoot playlist mismatches, inspect `spotify_track_resolution` logs for unmatched entries and `scheduled_job_context` to confirm prompts.
- Tail logs:
```
aws logs tail /aws/lambda/playlistbot-handler --follow --format short --since 2d | awk '
  {
    # split at the first " {" (space + opening brace)
    p = index($0, " {")
    if (p > 0) {
      prefix = substr($0, 1, p-1)
      json   = substr($0, p+1)   # starts with "{"
      print prefix
      print json
    } else {
      print $0
    }
  }
' \
| jq -Rr '
    if startswith("{")
    then (fromjson)
    else .
    end
'
```

## Reference Notes
- Legacy novelty guard knobs (`window_days`, `max_tracks_per_artist`, overlap ratios, `history_entries`) remain in DynamoDB for backward compatibility but do not alter behavior. History exists purely for audit/analytics.
- Playlists created during scheduled runs automatically adopt the naming pattern `<config_name> - <current rotation theme>`. Manual API calls append a date suffix (e.g., `Base Name 23-01-2026`).
- When `mode=update`, playlist metadata (name + description) is updated to include the human-readable variation axis derived from the latest OpenAI response.
