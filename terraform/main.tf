provider "aws" {
  region = var.aws_region
}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda"
  output_path = "${path.module}/lambda.zip"
}

data "aws_caller_identity" "current" {}

resource "aws_dynamodb_table" "playlist_state" {
  name         = "playlistbot_state"
  billing_mode = "PAY_PER_REQUEST"

  hash_key = "playlist_id"

  attribute {
    name = "playlist_id"
    type = "S"
  }

  tags = var.tags
}

resource "aws_iam_role" "lambda" {
  name               = "${var.lambda_function_name}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
  tags               = var.tags
}

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

data "aws_iam_policy_document" "lambda_inline" {
  statement {
    sid     = "AllowLogs"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = ["*"]
  }

  statement {
    sid = "AllowParameterRead"
    actions = [
      "ssm:GetParameter",
      "ssm:PutParameter"
    ]
    resources = [
      aws_ssm_parameter.spotify_client_id.arn,
      aws_ssm_parameter.spotify_refresh_token.arn,
      aws_ssm_parameter.openai_api_key.arn,
      aws_ssm_parameter.request_api_key.arn
    ]
  }

  statement {
    sid = "AllowRefreshWrite"
    actions = [
      "ssm:PutParameter"
    ]
    resources = [
      "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/playlistbot/spotify/refresh_token"
    ]
  }

  statement {
    sid = "AllowPlaylistStateDynamo"
    actions = [
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:UpdateItem",
      "dynamodb:Scan"
    ]
    resources = [
      aws_dynamodb_table.playlist_state.arn
    ]
  }
}

resource "aws_iam_role_policy" "lambda_inline" {
  name   = "${var.lambda_function_name}-inline"
  role   = aws_iam_role.lambda.id
  policy = data.aws_iam_policy_document.lambda_inline.json
}

resource "aws_lambda_function" "playlist" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda.arn
  runtime       = "python3.11"
  handler       = "app.handler"
  memory_size   = var.lambda_memory_mb
  timeout       = var.lambda_timeout_seconds
  reserved_concurrent_executions = var.lambda_reserved_concurrency

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      PARAM_SPOTIFY_CLIENT_ID     = aws_ssm_parameter.spotify_client_id.name
      PARAM_SPOTIFY_REFRESH_TOKEN = aws_ssm_parameter.spotify_refresh_token.name
      PARAM_OPENAI_API_KEY        = aws_ssm_parameter.openai_api_key.name
      PARAM_REQUEST_API_KEY       = aws_ssm_parameter.request_api_key.name
      DEFAULT_MARKET              = "AU"
      DEFAULT_PLAYLIST_PUBLIC     = "false"
      HANDLER_TIMEOUT_SECONDS     = tostring(var.lambda_timeout_seconds)
    }
  }

  tags = var.tags
}

resource "aws_api_gateway_rest_api" "playlist" {
  name = "prompt-playlist-api"
  tags = var.tags
}

resource "aws_api_gateway_resource" "playlist" {
  rest_api_id = aws_api_gateway_rest_api.playlist.id
  parent_id   = aws_api_gateway_rest_api.playlist.root_resource_id
  path_part   = "playlist"
}

resource "aws_api_gateway_method" "playlist_post" {
  rest_api_id   = aws_api_gateway_rest_api.playlist.id
  resource_id   = aws_api_gateway_resource.playlist.id
  http_method   = "POST"
  authorization = "NONE"

  api_key_required = true
}

resource "aws_api_gateway_integration" "playlist_post" {
  rest_api_id = aws_api_gateway_rest_api.playlist.id
  resource_id = aws_api_gateway_resource.playlist.id
  http_method = aws_api_gateway_method.playlist_post.http_method

  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = "arn:aws:apigateway:${var.aws_region}:lambda:path/2015-03-31/functions/${aws_lambda_function.playlist.arn}/invocations"
}

resource "aws_api_gateway_deployment" "playlist" {
  rest_api_id = aws_api_gateway_rest_api.playlist.id

  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_method.playlist_post.id,
      aws_api_gateway_integration.playlist_post.id
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [aws_api_gateway_integration.playlist_post]
}

resource "aws_api_gateway_stage" "prod" {
  rest_api_id  = aws_api_gateway_rest_api.playlist.id
  deployment_id = aws_api_gateway_deployment.playlist.id
  stage_name    = "prod"
  tags          = var.tags
}

resource "random_password" "api_key" {
  length  = 32
  special = false
}

resource "aws_api_gateway_api_key" "playlist" {
  name   = "playlistbot-shared-key"
  value  = random_password.api_key.result
  enabled = true
  tags    = var.tags
}

resource "aws_api_gateway_usage_plan" "playlist" {
  name = "playlistbot-usage-plan"

  throttle_settings {
    burst_limit = 2
    rate_limit  = 1
  }

  quota_settings {
    limit  = 1000
    period = "DAY"
  }

  api_stages {
    api_id = aws_api_gateway_rest_api.playlist.id
    stage  = aws_api_gateway_stage.prod.stage_name
  }

  tags = var.tags
}

resource "aws_api_gateway_usage_plan_key" "playlist" {
  key_id        = aws_api_gateway_api_key.playlist.id
  key_type      = "API_KEY"
  usage_plan_id = aws_api_gateway_usage_plan.playlist.id
}

resource "aws_lambda_permission" "apigw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.playlist.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.playlist.execution_arn}/*/*"
}

resource "aws_cloudwatch_event_rule" "daily_regen" {
  name                = "playlistbot-daily-regen"
  description         = "Daily playlist regeneration trigger"
  schedule_expression = "cron(10 17 * * ? *)" # 03:10 Australia/Melbourne = 17:10 UTC previous day
  tags                = var.tags
}

resource "aws_cloudwatch_event_target" "daily_regen_lambda" {
  rule      = aws_cloudwatch_event_rule.daily_regen.name
  target_id = "playlistbot-lambda"
  arn       = aws_lambda_function.playlist.arn
  input     = jsonencode({ scheduled = true })
}

resource "aws_lambda_permission" "events" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.playlist.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_regen.arn
}

resource "aws_ssm_parameter" "spotify_client_id" {
  name        = "/playlistbot/spotify/client_id"
  description = "Spotify Client ID for playlist bot"
  type        = "SecureString"
  value       = "REPLACE_ME"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "spotify_refresh_token" {
  name        = "/playlistbot/spotify/refresh_token"
  description = "Spotify refresh token for playlist bot"
  type        = "SecureString"
  value       = "REPLACE_ME"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "openai_api_key" {
  name        = "/playlistbot/openai/api_key"
  description = "OpenAI API key for playlist bot"
  type        = "SecureString"
  value       = "REPLACE_ME"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}

resource "aws_ssm_parameter" "request_api_key" {
  name        = "/playlistbot/security/api_key"
  description = "Shared secret for API Gateway requests"
  type        = "SecureString"
  value       = "REPLACE_ME"
  tags        = var.tags

  lifecycle {
    ignore_changes = [value]
  }
}
