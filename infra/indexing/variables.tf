variable "aws_region" {
  description = "AWS region used by the indexing infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming and tagging."
  type        = string
  default     = "agentic-research-rag"
}

variable "environment" {
  description = "Deployment environment."
  type        = string
  default     = "production"
}

variable "container_image_tag" {
  description = "Immutable ECR image tag used by the indexing task."
  type        = string
}

variable "corpus_bucket_name" {
  description = "S3 bucket containing the source PDF corpus."
  type        = string
  default     = "agentic-research-rag-corpus-649616336579"
}

variable "corpus_prefix" {
  description = "S3 prefix containing the source PDF corpus."
  type        = string
  default     = "source-documents"
}

variable "artifact_bucket_name" {
  description = "S3 bucket used for generated retrieval artifacts."
  type        = string
  default     = "agentic-research-rag"
}

variable "artifact_prefix" {
  description = "S3 prefix used for generated retrieval artifacts."
  type        = string
  default     = "agentic-research-rag"
}

variable "indexer_cpu" {
  description = "CPU units allocated to the Fargate indexing task."
  type        = number
  default     = 2048
}

variable "indexer_memory" {
  description = "Memory in MiB allocated to the Fargate indexing task."
  type        = number
  default     = 4096
}

variable "indexer_task_role_arn" {
  description = "IAM role assumed by the Fargate indexing task."
  type        = string
}