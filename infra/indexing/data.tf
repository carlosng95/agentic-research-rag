data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_ecr_repository" "app" {
  name = "agentic_research_rag"
}

data "aws_ecr_image" "indexer" {
  repository_name = data.aws_ecr_repository.app.name
  image_tag       = var.container_image_tag
}

data "aws_iam_role" "ecs_execution" {
  name = "agentic-rag-ecs-execution-role"
}

data "aws_cloudwatch_log_group" "ecs" {
  name = "/ecs/agentic-research-rag"
}
