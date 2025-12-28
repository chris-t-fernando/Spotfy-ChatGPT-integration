# Prompt → Spotify Playlist Lambda

Python 3.11 AWS Lambda that turns a natural-language prompt into a curated Spotify playlist through OpenAI + Spotify APIs. Terraform in `../terraform` provisions the Lambda, IAM role, API Gateway HTTP API, and SSM parameters required for secrets.

## Architecture Overview
- **API Gateway HTTP API** exposes `POST /playlist` with an `x-api-key` header enforced in the Lambda.
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

## Calling the Endpoint
Replace `<API_URL>` and `<API_KEY>` with your deployment outputs.

**Create new playlist**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "give me chill synthwave for late-night focus"
  }'
```

**Update existing playlist**
```bash
curl -X POST "<API_URL>/playlist" \
  -H "content-type: application/json" \
  -H "x-api-key: <API_KEY>" \
  -d '{
    "prompt": "amp me up for a tempo run",
    "playlist_name": "Tempo Run Booster"
  }'
```
If `playlist_name` matches an existing playlist (case-insensitive exact match), the Lambda clears all items and replaces them with the generated tracks without renaming the playlist.

Responses include:
```json
{
  "status": "done",
  "message": "done, the playlist name is \"Tempo Run Booster\"",
  "playlist_name": "Tempo Run Booster",
  "playlist_url": "https://open.spotify.com/playlist/…",
  "matched": 48,
  "unmatched": ["Artist – Song"]
}
```

## Notes
- All outbound HTTP calls have 10s connect / 30s read timeouts.
- SSM values are cached between invocations for performance.
- Logs include `[info]`, `[warning]`, and `[critical]` prefixes for easier CloudWatch filtering.
