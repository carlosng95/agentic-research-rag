resource "aws_cloudwatch_metric_alarm" "research_request_errors" {
  alarm_name        = "${var.project_name}-${var.environment}-research-errors"
  alarm_description = "Research endpoint returned one or more server errors."

  namespace   = local.metrics_namespace
  metric_name = "RequestErrorCount"

  dimensions = {
    Service     = local.metrics_service
    Environment = var.environment
    Endpoint    = "/research"
    Method      = "POST"
  }

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions             = []
  ok_actions                = []
  insufficient_data_actions = []
}


resource "aws_cloudwatch_metric_alarm" "alb_unhealthy_target" {
  alarm_name        = "${var.project_name}-${var.environment}-alb-unhealthy-target"
  alarm_description = "The application load balancer has no healthy ECS target."

  namespace   = "AWS/ApplicationELB"
  metric_name = "HealthyHostCount"

  dimensions = {
    LoadBalancer = aws_lb.app.arn_suffix
    TargetGroup  = aws_lb_target_group.app.arn_suffix
  }

  statistic           = "Minimum"
  period              = 60
  evaluation_periods  = 2
  datapoints_to_alarm = 2

  threshold           = 1
  comparison_operator = "LessThanThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions             = []
  ok_actions                = []
  insufficient_data_actions = []
}


resource "aws_cloudwatch_metric_alarm" "ecs_memory_high" {
  alarm_name        = "${var.project_name}-${var.environment}-ecs-memory-high"
  alarm_description = "ECS service memory utilization remains above 85 percent."

  namespace   = "AWS/ECS"
  metric_name = "MemoryUtilization"

  dimensions = {
    ClusterName = aws_ecs_cluster.app.name
    ServiceName = aws_ecs_service.app.name
  }

  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2

  threshold           = 85
  comparison_operator = "GreaterThanThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions             = []
  ok_actions                = []
  insufficient_data_actions = []
}


resource "aws_cloudwatch_metric_alarm" "cross_encoder_latency_high" {
  alarm_name        = "${var.project_name}-${var.environment}-cross-encoder-latency-high"
  alarm_description = "Cross-encoder reranking latency is consistently above 5 seconds."

  namespace   = local.metrics_namespace
  metric_name = "OperationLatency"

  dimensions = {
    Service     = local.metrics_service
    Environment = var.environment
    Operation   = "cross_encoder_scoring"
  }

  statistic           = "Average"
  period              = 300
  evaluation_periods  = 3
  datapoints_to_alarm = 2

  threshold           = 5000
  comparison_operator = "GreaterThanThreshold"

  treat_missing_data = "notBreaching"

  alarm_actions             = []
  ok_actions                = []
  insufficient_data_actions = []
}