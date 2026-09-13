variable "aws_region" {
  description = "AWS region used by the project"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Base name used for project resources"
  type        = string
  default     = "agentic-research-rag"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "production"
}

variable "alb_ingress_cidr" {
  description = "CIDR allowed to access the public ALB over HTTP."
  type        = string

  validation {
    condition     = can(cidrnetmask(var.alb_ingress_cidr))
    error_message = "alb_ingress_cidr must be a valid IPv4 CIDR block."
  }
}

variable "container_port" {
  description = "Port exposed by the FastAPI container."
  type        = number
  default     = 8000
}

variable "container_image_tag" {
  description = "Immutable ECR image tag deployed by ECS."
  type        = string
  default     = "3181beb"
}

variable "task_cpu" {
  description = "CPU units allocated to the Fargate task."
  type        = number
  default     = 2048
}

variable "task_memory" {
  description = "Memory allocated to the Fargate task in MiB."
  type        = number
  default     = 4096
}

variable "service_desired_count" {
  description = "Number of ECS tasks kept running by the service."
  type        = number
  default     = 1

  validation {
    condition     = var.service_desired_count >= 0
    error_message = "service_desired_count must be zero or greater."
  }
}