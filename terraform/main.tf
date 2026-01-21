terraform {
  required_providers {
    pinecone = {
      source = "pinecone-io/pinecone"
      version = "0.2.0"
    }
  }
}

provider "pinecone" {
  api_key = var.pinecone_api_key
}

resource "pinecone_index" "ikms_knowledge_base" {
  name      = "ikms-rag-index"
  dimension = 1536
  metric    = "cosine"
  spec = {
    serverless = {
      cloud  = "aws"
      region = "us-east-1"
    }
  }
}

variable "pinecone_api_key" {
  type      = string
  sensitive = true
}