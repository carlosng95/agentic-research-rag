resource "aws_security_group" "alb" {
  name        = "agentic-rag-alb-sg"
  description = "Security group for the Agentic Research RAG Application Load Balancer."
  vpc_id      = data.aws_vpc.default.id

  tags = {
    Name = "agentic-rag-alb-sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "alb_http" {
  security_group_id = aws_security_group.alb.id

  description = "Allow HTTP traffic from the configured client CIDR."

  cidr_ipv4   = var.alb_ingress_cidr
  from_port   = 80
  to_port     = 80
  ip_protocol = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "alb_all" {
  security_group_id = aws_security_group.alb.id

  description = "Allow outbound traffic from the ALB."

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}

resource "aws_security_group" "fargate" {
  name        = "agentic-rag-fargate-sg"
  description = "Security group for the Agentic Research RAG Fargate tasks."
  vpc_id      = data.aws_vpc.default.id

  tags = {
    Name = "agentic-rag-fargate-sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "fargate_from_alb" {
  security_group_id = aws_security_group.fargate.id

  description = "Allow application traffic only from the ALB."

  referenced_security_group_id = aws_security_group.alb.id
  from_port                    = var.container_port
  to_port                      = var.container_port
  ip_protocol                  = "tcp"
}

resource "aws_vpc_security_group_egress_rule" "fargate_all" {
  security_group_id = aws_security_group.fargate.id

  description = "Allow outbound traffic from Fargate tasks."

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"
}