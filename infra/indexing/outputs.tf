output "ecs_cluster_name" {
  description = "ECS cluster used for one-off indexing tasks."
  value       = aws_ecs_cluster.indexing.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS indexing cluster."
  value       = aws_ecs_cluster.indexing.arn
}

output "task_definition_arn" {
  description = "ARN of the indexing task definition."
  value       = aws_ecs_task_definition.indexer.arn
}

output "security_group_id" {
  description = "Security group used by indexing tasks."
  value       = aws_security_group.indexer.id
}

output "subnet_ids" {
  description = "Default VPC subnets available to indexing tasks."
  value       = data.aws_subnets.default.ids
}

output "container_image_tag" {
  description = "Git SHA used to resolve the indexing container image."
  value       = var.container_image_tag
}

output "container_image_digest" {
  description = "Immutable ECR digest used by the indexing task definition."
  value       = data.aws_ecr_image.indexer.image_digest
}

output "indexer_task_role_arn" {
  description = "IAM role assumed by the indexing application."
  value       = var.indexer_task_role_arn
}
