resource "aws_ecs_cluster" "app" {
  name = "agentic-research-rag-cluster"

  tags = {
    Name = "agentic-research-rag-cluster"
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = "agentic-research-rag"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"

  cpu    = var.task_cpu
  memory = var.task_memory

  task_role_arn      = data.aws_iam_role.ecs_task.arn
  execution_role_arn = data.aws_iam_role.ecs_execution.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "ARM64"
  }

  container_definitions = jsonencode([
    {
      name      = "agentic-research-rag"
      image     = "${data.aws_ecr_repository.app.repository_url}@${data.aws_ecr_image.app.image_digest}"
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
          protocol      = "tcp"
        }
      ]

      environmentFiles = [
        {
          type  = "s3"
          value = "${data.aws_s3_bucket.artifacts.arn}/config/production.env"
        }
      ]

      secrets = [
        {
          name      = "OPENAI_API_KEY"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:OPENAI_API_KEY::"
        },
        {
          name      = "TAVILY_API_KEY"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:TAVILY_API_KEY::"
        },
        {
          name      = "GOOGLE_API_KEY"
          valueFrom = "${data.aws_secretsmanager_secret.api_keys.arn}:GOOGLE_API_KEY::"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"

        options = {
          "awslogs-group"         = data.aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }
    }
  ])

  tags = {
    Name = "agentic-research-rag"
  }
}

resource "aws_ecs_service" "app" {
  name            = "agentic-rag-service"
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn

  desired_count = var.service_desired_count
  launch_type   = "FARGATE"

  platform_version = "LATEST"

  health_check_grace_period_seconds = 180

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = data.aws_subnets.default.ids
    security_groups  = [aws_security_group.fargate.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "agentic-research-rag"
    container_port   = var.container_port
  }

  depends_on = [
    aws_lb_listener.http
  ]

  tags = {
    Name = "agentic-rag-service"
  }
}