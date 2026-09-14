output "aws_account_id" {
  description = "AWS account used by Terraform."
  value       = data.aws_caller_identity.current.account_id
}

output "artifact_bucket_name" {
  description = "Existing S3 artifact bucket."
  value       = data.aws_s3_bucket.artifacts.id
}

output "ecr_repository_url" {
  description = "Existing ECR repository URL."
  value       = data.aws_ecr_repository.app.repository_url
}

output "api_keys_secret_arn" {
  description = "ARN of the existing application API keys secret."
  value       = data.aws_secretsmanager_secret.api_keys.arn
}

output "ecs_task_role_arn" {
  description = "Existing ECS application task role."
  value       = data.aws_iam_role.ecs_task.arn
}

output "ecs_execution_role_arn" {
  description = "Existing ECS task execution role."
  value       = data.aws_iam_role.ecs_execution.arn
}

output "ecs_log_group_name" {
  description = "Existing ECS CloudWatch log group."
  value       = data.aws_cloudwatch_log_group.ecs.name
}

output "vpc_id" {
  description = "Default VPC used by the runtime."
  value       = data.aws_vpc.default.id
}

output "subnet_ids" {
  description = "Default subnets available to the runtime."
  value       = data.aws_subnets.default.ids
}

output "alb_security_group_id" {
  description = "Security group used by the Application Load Balancer."
  value       = aws_security_group.alb.id
}

output "fargate_security_group_id" {
  description = "Security group used by the ECS Fargate tasks."
  value       = aws_security_group.fargate.id
}

output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer."
  value       = aws_lb.app.dns_name
}

output "alb_arn" {
  description = "ARN of the Application Load Balancer."
  value       = aws_lb.app.arn
}

output "target_group_arn" {
  description = "ARN of the ECS application target group."
  value       = aws_lb_target_group.app.arn
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster."
  value       = aws_ecs_cluster.app.arn
}

output "ecs_task_definition_arn" {
  description = "ARN of the ECS task definition."
  value       = aws_ecs_task_definition.app.arn
}

output "container_image_digest" {
  description = "Digest of the ECR image deployed by ECS."
  value       = data.aws_ecr_image.app.image_digest
}

output "ecs_service_name" {
  description = "Name of the ECS service."
  value       = aws_ecs_service.app.name
}

output "ecs_service_id" {
  description = "ID of the ECS service."
  value       = aws_ecs_service.app.id
}

output "cloudwatch_dashboard_name" {
  description = "CloudWatch observability dashboard name."
  value       = aws_cloudwatch_dashboard.observability.dashboard_name
}

output "cloudwatch_alarm_names" {
  description = "CloudWatch observability alarm names."

  value = [
    aws_cloudwatch_metric_alarm.research_request_errors.alarm_name,
    aws_cloudwatch_metric_alarm.alb_unhealthy_target.alarm_name,
    aws_cloudwatch_metric_alarm.ecs_memory_high.alarm_name,
    aws_cloudwatch_metric_alarm.cross_encoder_latency_high.alarm_name,
  ]
}