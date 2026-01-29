# =============================================================================
# Arquivo: services/bedrock_kb_service.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Finalidade:
#   Recuperação de conhecimento (RAG) via Amazon Bedrock Knowledge Bases.
#   A Knowledge Base é configurada no Console apontando para S3 Vectors.
#   Este serviço executa "retrieve" e normaliza para evidências consumíveis no prompt.
# =============================================================================

import boto3
from botocore.config import Config as BotoConfig
from typing import Dict, Any, List, Optional


class BedrockKnowledgeBaseService:
    def __init__(self, region: str, timeout_seconds: int):
        self.client = boto3.client(
            "bedrock-agent-runtime",
            region_name=region,
            config=BotoConfig(
                read_timeout=timeout_seconds,
                connect_timeout=timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            ),
        )

    def retrieve(
        self,
        knowledge_base_id: str,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recupera evidências via Knowledge Base.
        Para MVP, filtros são opcionais e dependem do schema de metadados definido na KB.
        """
        req: Dict[str, Any] = {
            "knowledgeBaseId": knowledge_base_id,
            "retrievalQuery": {"text": query_text},
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k
                }
            },
        }

        if filters:
            req["retrievalConfiguration"]["vectorSearchConfiguration"]["filter"] = filters

        return self.client.retrieve(**req)

    @staticmethod
    def normalize(resp: Dict[str, Any], score_threshold: float) -> Dict[str, Any]:
        """
        Converte o retorno do retrieve para um formato simples de evidências.
        """
        results = resp.get("retrievalResults", [])
        evidences: List[Dict[str, Any]] = []
        citations: List[str] = []

        best_score = 0.0
        for r in results:
            score = float(r.get("score", 0.0))
            best_score = max(best_score, score)

            content = r.get("content", {})
            snippet = content.get("text", "")[:500]

            location = r.get("location", {})
            doc_id = str(location.get("s3Location", {}).get("uri", "")) or str(location)
            chunk_id = r.get("metadata", {}).get("chunkId", "desconhecido")

            evidences.append({
                "docId": doc_id,
                "chunkId": chunk_id,
                "score": score,
                "snippet": snippet,
            })
            citations.append(f"{doc_id}:{chunk_id}")

        flags: List[str] = []
        no_evidence = (len(evidences) == 0) or (best_score < score_threshold)
        if no_evidence:
            flags.append("no_evidence")

        return {
            "evidences": evidences,
            "citations": citations,
            "flags": flags,
            "no_evidence": no_evidence,
        }
