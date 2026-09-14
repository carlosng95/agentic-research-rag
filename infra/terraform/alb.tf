resource "aws_lb" "app" {
  name               = "agentic-rag-alb"
  internal           = false
  load_balancer_type = "application"

  security_groups = [aws_security_group.alb.id]
  subnets         = data.aws_subnets.default.ids

  enable_deletion_protection = false

  tags = {
    Name = "agentic-rag-alb"
  }
}

resource "aws_lb_target_group" "app" {
  name        = "agentic-rag-tg"
  port        = var.container_port
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = data.aws_vpc.default.id

  deregistration_delay = 30

  health_check {
    enabled             = true
    path                = "/ready"
    protocol            = "HTTP"
    port                = "traffic-port"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }

  tags = {
    Name = "agentic-rag-tg"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.app.arn

  port     = 80
  protocol = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}