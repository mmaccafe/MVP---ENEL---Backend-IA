# =============================================================================
# Arquivo: services/bedrock_runtime_service.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Invocação de modelo no Amazon Bedrock Runtime.
# Autenticação:
#   - Não existe token do Bedrock para aplicação.
#   - A autenticação é IAM (SigV4) e o boto3 assina a requisição.
#   - Em ECS, usar Task Role.
# =============================================================================

import json
import boto3
from botocore.config import Config as BotoConfig
from typing import Dict, Any


class BedrockRuntimeService:
    def __init__(self, region: str, timeout_seconds: int):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=BotoConfig(
                read_timeout=timeout_seconds,
                connect_timeout=timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    def invoke_text_model(self, model_id: str, prompt: str, generation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoca modelo de texto.
        Observação:
          O body varia por provedor/modelo. Este corpo é propositalmente "genérico"
          para MVP e deve ser adaptado conforme o modelo selecionado.
        """
        body = {
            "prompt": prompt,
            "max_tokens": generation.get("maxOutputTokens"),
            "temperature": generation.get("temperature"),
            "top_p": generation.get("topP"),
        }

        resp = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(body).encode("utf-8"),
            accept="application/json",
            contentType="application/json",
        )

        raw = resp["body"].read().decode("utf-8")
        data = json.loads(raw)

        # Normalização mínima (ajustar conforme provedor)
        reply_text = data.get("completion") or data.get("outputText") or str(data)

        return {
            "replyText": reply_text,
            "raw": data,
            "modelId": model_id,
        }
