variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "ap-southeast-2"
}

variable "lambda_function_name" {
  description = "Name of the playlist Lambda function"
  type        = string
  default     = "playlistbot-handler"
}

variable "lambda_memory_mb" {
  description = "Memory size for the Lambda function"
  type        = number
  default     = 1024
}

variable "lambda_timeout_seconds" {
  description = "Execution timeout for the Lambda function"
  type        = number
  default     = 600
}

variable "lambda_reserved_concurrency" {
  description = "Reserved concurrency for the Lambda function"
  type        = number
  default     = 1
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default     = {
    Project = "prompt-playlist"
  }
}
