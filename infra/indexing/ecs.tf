resource "aws_ecs_cluster" "indexing" {
  name = "${var.project_name}-indexing-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled"
  }

  tags = {
    Name = "${var.project_name}-indexing-cluster"
  }
}

resource "aws_ecs_task_definition" "indexer" {
  family                   = "${var.project_name}-indexer"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = tostring(var.indexer_cpu)
  memory                   = tostring(var.indexer_memory)
  execution_role_arn       = data.aws_iam_role.ecs_execution.arn
  task_role_arn            = var.indexer_task_role_arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }

  container_definitions = jsonencode([
    {
      name      = "${var.project_name}-indexer"
      image     = "${data.aws_ecr_repository.app.repository_url}@${data.aws_ecr_image.indexer.image_digest}"
      essential = true

      command = [
        "python",
        "-m",
        "agentic_research_rag.indexing.job"
      ]

      environment = [
        {
          name  = "AWS_REGION"
          value = var.aws_region
        },
        {
          name  = "CORPUS_SOURCE"
          value = "s3"
        },
        {
          name  = "CORPUS_BUCKET"
          value = var.corpus_bucket_name
        },
        {
          name  = "CORPUS_PREFIX"
          value = var.corpus_prefix
        },
        {
          name  = "ARTIFACT_BUCKET"
          value = var.artifact_bucket_name
        },
        {
          name  = "ARTIFACT_PREFIX"
          value = var.artifact_prefix
        },
        {
          name  = "EMBEDDING_BACKEND"
          value = "local"
        },
        {
          name  = "CHUNK_SIZE"
          value = "1200"
        },
        {
          name  = "CHUNK_OVERLAP"
          value = "200"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"

        options = {
          awslogs-group         = data.aws_cloudwatch_log_group.ecs.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "indexer"
        }
      }
    }
  ])

  tags = {
    Name = "${var.project_name}-indexer"
  }
}
