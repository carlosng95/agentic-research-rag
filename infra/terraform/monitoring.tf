locals {
  metrics_namespace = "AgenticResearchRAG"
  metrics_service   = "agentic-research-rag"
  metrics_model     = "gpt-4o-mini"
}

resource "aws_cloudwatch_dashboard" "observability" {
  dashboard_name = "${var.project_name}-${var.environment}-observability"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "text"
        x      = 0
        y      = 0
        width  = 24
        height = 2

        properties = {
          markdown = <<-EOT
            # Agentic Research RAG — Production Observability

            Application, RAG, retrieval and LLM telemetry for the production service.
          EOT
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 2
        width  = 8
        height = 6

        properties = {
          title   = "Research Requests"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Sum"

          metrics = [
            [
              local.metrics_namespace,
              "ResearchRequestCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                label = "Requests"
              }
            ]
          ]
        }
      },

      {
        type   = "metric"
        x      = 8
        y      = 2
        width  = 8
        height = 6

        properties = {
          title   = "Research Latency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              local.metrics_namespace,
              "ResearchLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                stat  = "p50"
                label = "p50"
              }
            ],
            [
              "...",
              {
                stat  = "p95"
                label = "p95"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Milliseconds"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 16
        y      = 2
        width  = 8
        height = 6

        properties = {
          title   = "Research Errors"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Sum"

          metrics = [
            [
              local.metrics_namespace,
              "ResearchErrorCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                label = "Research errors"
              }
            ],
            [
              local.metrics_namespace,
              "CitationInvalidCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                label = "Invalid citations"
              }
            ]
          ]
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 8
        width  = 8
        height = 6

        properties = {
          title   = "Paper Evidence Sufficiency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              local.metrics_namespace,
              "PaperEvidenceSufficient",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                stat  = "Average"
                label = "Sufficiency ratio"
              }
            ]
          ]

          yAxis = {
            left = {
              min       = 0
              max       = 1
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 8
        y      = 8
        width  = 8
        height = 6

        properties = {
          title   = "Web Fallbacks"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Sum"

          metrics = [
            [
              local.metrics_namespace,
              "WebFallbackCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                label = "Web fallbacks"
              }
            ]
          ]
        }
      },

      {
        type   = "metric"
        x      = 16
        y      = 8
        width  = 8
        height = 6

        properties = {
          title   = "Retrieved Documents"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Average"

          metrics = [
            [
              local.metrics_namespace,
              "RetrievedDocumentCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              {
                label = "Documents"
              }
            ]
          ]
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 14
        width  = 12
        height = 7

        properties = {
          title   = "LLM Latency by Operation"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              local.metrics_namespace,
              "LLMCallLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "query_rewrite",
              "Model",
              local.metrics_model,
              {
                stat  = "Average"
                label = "query rewrite"
              }
            ],
            [
              local.metrics_namespace,
              "LLMCallLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "paper_generation",
              "Model",
              local.metrics_model,
              {
                stat  = "Average"
                label = "paper generation"
              }
            ],
            [
              local.metrics_namespace,
              "LLMCallLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "sufficiency",
              "Model",
              local.metrics_model,
              {
                stat  = "Average"
                label = "sufficiency"
              }
            ],
            [
              local.metrics_namespace,
              "LLMCallLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "synthesis",
              "Model",
              local.metrics_model,
              {
                stat  = "Average"
                label = "synthesis"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Milliseconds"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 12
        y      = 14
        width  = 12
        height = 7

        properties = {
          title   = "LLM Token Consumption"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Sum"

          metrics = [
            [
              local.metrics_namespace,
              "LLMTotalTokens",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "query_rewrite",
              "Model",
              local.metrics_model,
              {
                label = "query rewrite"
              }
            ],
            [
              local.metrics_namespace,
              "LLMTotalTokens",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "paper_generation",
              "Model",
              local.metrics_model,
              {
                label = "paper generation"
              }
            ],
            [
              local.metrics_namespace,
              "LLMTotalTokens",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "sufficiency",
              "Model",
              local.metrics_model,
              {
                label = "sufficiency"
              }
            ],
            [
              local.metrics_namespace,
              "LLMTotalTokens",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "synthesis",
              "Model",
              local.metrics_model,
              {
                label = "synthesis"
              }
            ]
          ]
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 21
        width  = 12
        height = 7

        properties = {
          title   = "Retrieval Latency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Average"

          metrics = [
            [
              local.metrics_namespace,
              "OperationLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "semantic_retrieval",
              {
                label = "semantic retrieval"
              }
            ],
            [
              local.metrics_namespace,
              "OperationLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "bm25_retrieval",
              {
                label = "BM25 retrieval"
              }
            ],
            [
              local.metrics_namespace,
              "OperationLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "cross_encoder_scoring",
              {
                label = "cross encoder"
              }
            ],
            [
              local.metrics_namespace,
              "OperationLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Operation",
              "paper_retrieval",
              {
                label = "complete paper retrieval"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Milliseconds"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 12
        y      = 21
        width  = 12
        height = 7

        properties = {
          title   = "Graph Node Latency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Average"

          metrics = [
            [
              local.metrics_namespace,
              "GraphNodeLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Node",
              "rewrite_query",
              {
                label = "rewrite query"
              }
            ],
            [
              local.metrics_namespace,
              "GraphNodeLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Node",
              "paper_rag",
              {
                label = "paper RAG"
              }
            ],
            [
              local.metrics_namespace,
              "GraphNodeLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Node",
              "evaluate",
              {
                label = "evaluate"
              }
            ],
            [
              local.metrics_namespace,
              "GraphNodeLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Node",
              "web_search",
              {
                label = "web search"
              }
            ],
            [
              local.metrics_namespace,
              "GraphNodeLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Node",
              "synthesize",
              {
                label = "synthesize"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Milliseconds"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 28
        width  = 12
        height = 7

        properties = {
          title   = "HTTP /research Latency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              local.metrics_namespace,
              "RequestLatency",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Endpoint",
              "/research",
              "Method",
              "POST",
              {
                stat  = "p50"
                label = "p50"
              }
            ],
            [
              "...",
              {
                stat  = "p95"
                label = "p95"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Milliseconds"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 12
        y      = 28
        width  = 12
        height = 7

        properties = {
          title   = "HTTP Request Errors"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300
          stat    = "Sum"

          metrics = [
            [
              local.metrics_namespace,
              "RequestErrorCount",
              "Service",
              local.metrics_service,
              "Environment",
              var.environment,
              "Endpoint",
              "/research",
              "Method",
              "POST",
              {
                label = "/research errors"
              }
            ]
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 35
        width  = 12
        height = 7

        properties = {
          title   = "ECS CPU Utilization"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              "AWS/ECS",
              "CPUUtilization",
              "ClusterName",
              aws_ecs_cluster.app.name,
              "ServiceName",
              aws_ecs_service.app.name,
              {
                stat  = "Average"
                label = "Average CPU"
              }
            ],
            [
              "...",
              {
                stat  = "Maximum"
                label = "Maximum CPU"
              }
            ]
          ]

          yAxis = {
            left = {
              min       = 0
              max       = 100
              label     = "Percent"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 12
        y      = 35
        width  = 12
        height = 7

        properties = {
          title   = "ECS Memory Utilization"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              "AWS/ECS",
              "MemoryUtilization",
              "ClusterName",
              aws_ecs_cluster.app.name,
              "ServiceName",
              aws_ecs_service.app.name,
              {
                stat  = "Average"
                label = "Average memory"
              }
            ],
            [
              "...",
              {
                stat  = "Maximum"
                label = "Maximum memory"
              }
            ]
          ]

          yAxis = {
            left = {
              min       = 0
              max       = 100
              label     = "Percent"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 0
        y      = 42
        width  = 12
        height = 7

        properties = {
          title   = "ALB Traffic and Latency"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              "AWS/ApplicationELB",
              "RequestCount",
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "Sum"
                label = "Requests"
                yAxis = "right"
              }
            ],
            [
              "AWS/ApplicationELB",
              "TargetResponseTime",
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "Average"
                label = "Average response time"
              }
            ],
            [
              "AWS/ApplicationELB",
              "TargetResponseTime",
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "p95"
                label = "p95 response time"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Seconds"
              showUnits = false
            }

            right = {
              label     = "Requests"
              showUnits = false
            }
          }
        }
      },

      {
        type   = "metric"
        x      = 12
        y      = 42
        width  = 12
        height = 7

        properties = {
          title   = "ALB Health and 5xx Errors"
          region  = var.aws_region
          view    = "timeSeries"
          stacked = false
          period  = 300

          metrics = [
            [
              "AWS/ApplicationELB",
              "HTTPCode_Target_5XX_Count",
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "Sum"
                label = "Target 5xx"
              }
            ],
            [
              "AWS/ApplicationELB",
              "HTTPCode_ELB_5XX_Count",
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "Sum"
                label = "ALB 5xx"
              }
            ],
            [
              "AWS/ApplicationELB",
              "HealthyHostCount",
              "TargetGroup",
              aws_lb_target_group.app.arn_suffix,
              "LoadBalancer",
              aws_lb.app.arn_suffix,
              {
                stat  = "Minimum"
                label = "Healthy targets"
                yAxis = "right"
              }
            ]
          ]

          yAxis = {
            left = {
              label     = "Errors"
              showUnits = false
            }

            right = {
              min       = 0
              label     = "Healthy targets"
              showUnits = false
            }
          }
        }
      }
    ]
  })
}