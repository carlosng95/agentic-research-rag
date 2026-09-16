resource "aws_security_group" "indexer" {
  name        = "${var.project_name}-indexer-sg"
  description = "Security group for one-off Fargate indexing tasks."
  vpc_id      = data.aws_vpc.default.id

  tags = {
    Name = "${var.project_name}-indexer-sg"
  }
}

resource "aws_vpc_security_group_egress_rule" "indexer_all" {
  security_group_id = aws_security_group.indexer.id
  description       = "Allow indexing tasks to reach AWS services and external model repositories."

  ip_protocol = "-1"
  cidr_ipv4   = "0.0.0.0/0"
}
