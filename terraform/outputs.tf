output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.playlist.function_name
}

output "rest_api_invoke_url" {
  description = "Invoke URL for the REST API"
  value       = "https://${aws_api_gateway_rest_api.playlist.id}.execute-api.${var.aws_region}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}"
}

output "api_key_value" {
  description = "Provisioned API key value"
  value       = aws_api_gateway_api_key.playlist.value
  sensitive   = true
}

output "ssm_parameters" {
  description = "SecureString parameters that require production values"
  value = [
    aws_ssm_parameter.spotify_client_id.name,
    aws_ssm_parameter.spotify_refresh_token.name,
    aws_ssm_parameter.openai_api_key.name,
    aws_ssm_parameter.request_api_key.name
  ]
}
