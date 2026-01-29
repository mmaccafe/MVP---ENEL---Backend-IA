# =============================================================================
# Arquivo: config.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Centraliza configurações via variáveis de ambiente para execução em:
#   - Desenvolvimento local (AWS CLI/SSO)
#   - Container (ECS/Fargate com IAM Task Role)
# =============================================================================

import os


class Config:
    # AWS / Região
    AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

    # Bedrock Runtime - modelos por endpoint (ajustar conforme padrão do projeto)
    BEDROCK_TURN_MODEL_ID = os.getenv(
        "BEDROCK_TURN_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0"
    )
    BEDROCK_EVALUATE_MODEL_ID = os.getenv(
        "BEDROCK_EVALUATE_MODEL_ID",
        "anthropic.claude-3-sonnet-20240229-v1:0"
    )
    BEDROCK_SUMMARIZE_MODEL_ID = os.getenv(
        "BEDROCK_SUMMARIZE_MODEL_ID",
        "anthropic.claude-3-haiku-20240307-v1:0"
    )

    # Bedrock Knowledge Base (RAG) - configurada no Console usando S3 Vectors
    BEDROCK_KB_ID = os.getenv("BEDROCK_KB_ID", "")

    # Templates no bundle do container (Git -> build -> imagem ECS)
    TEMPLATES_ROOT = os.getenv("TEMPLATES_ROOT", "./templates")

    # Timeouts e retries
    BEDROCK_TIMEOUT_SECONDS = int(os.getenv("BEDROCK_TIMEOUT_SECONDS", "20"))

    # Defaults de geração (podem ser sobrescritos no payload)
    DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "280"))
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))

    # Defaults de RAG
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
    DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.80"))
