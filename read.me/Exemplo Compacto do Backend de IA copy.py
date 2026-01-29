"""
Exemplo de API Flask para: POST /v1/ai/turn

Responsabilidades deste serviço (Backend de IA):
- Receber um Turn já preparado pelo Backend de Aplicação
- Executar RAG (S3 Vector)
- Compor prompt versionado
- Executar exatamente uma inferência no Amazon Bedrock
- Aplicar guardrails e pós-processamento
- Retornar resposta textual + metadados


"""

import os
import time
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# ============================================================
# Configurações gerais
# ============================================================

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Modelo Bedrock utilizado para Turn conversacional
MODEL_ID_TURN = os.getenv(
    "MODEL_ID_TURN",
    "anthropic.claude-3-haiku-20240307-v1:0"
)

# Parâmetros de inferência (respostas curtas e controladas)
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "280"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Parâmetros padrão de RAG
DEFAULT_TOPK = int(os.getenv("DEFAULT_TOPK", "5"))
DEFAULT_SCORE_THRESHOLD = float(os.getenv("DEFAULT_SCORE_THRESHOLD", "0.75"))

# Limites defensivos de payload
MAX_USER_TEXT_CHARS = int(os.getenv("MAX_USER_TEXT_CHARS", "2000"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "4000"))
MAX_LAST_TURNS = int(os.getenv("MAX_LAST_TURNS", "3"))
MAX_SNIPPET_CHARS = int(os.getenv("MAX_SNIPPET_CHARS", "800"))

# Templates de prompt versionados
# (em produção: Git, S3 ou tabela no Postgres)
PROMPT_TEMPLATES: Dict[str, str] = {
    "turn-default-v1": (
        "SYSTEM:\n"
        "Você é um médico simulado para treinamento. "
        "Seja conciso e objetivo. "
        "Não invente fatos clínicos.\n\n"
        "CONTRATO DE SAÍDA:\n"
        "- Responda em pt-BR\n"
        "- Use frases curtas\n"
        "- Priorize perguntas para coletar informações faltantes\n\n"
        "CENÁRIO:\n{scenario}\n\n"
        "RESUMO DA CONVERSA:\n{summary}\n\n"
        "ÚLTIMAS INTERAÇÕES:\n{last_turns}\n\n"
        "EVIDÊNCIAS:\n{evidence}\n\n"
        "USUÁRIO:\n{user}\n\n"
        "ASSISTENTE:\n"
    )
}

# Cliente Bedrock Runtime
bedrock = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config(
        read_timeout=int(os.getenv("BEDROCK_READ_TIMEOUT", "25")),
        connect_timeout=int(os.getenv("BEDROCK_CONNECT_TIMEOUT", "5")),
        retries={"max_attempts": int(os.getenv("BEDROCK_MAX_ATTEMPTS", "2"))},
    ),
)

app = Flask(__name__)

# ============================================================
# Funções utilitárias
# ============================================================

def normalizar_texto(texto: str, limite: int) -> str:
    """
    Normaliza texto para uso interno:
    - remove caracteres de controle
    - normaliza espaços
    - aplica truncamento defensivo
    """
    texto = (texto or "").strip()
    texto = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto[:limite]

def exigir_header(nome: str) -> str:
    """
    Garante que um header obrigatório esteja presente.
    """
    valor = request.headers.get(nome)
    if not valor:
        raise ValueError(f"Header obrigatório ausente: {nome}")
    return valor

def validar_payload(body: Dict[str, Any]) -> None:
    """
    Valida estrutura mínima do payload do Turn.
    """
    obrigatorios = [
        "sessionId",
        "turnId",
        "userText",
        "conversationSummary",
        "scenarioContext"
    ]
    for campo in obrigatorios:
        if campo not in body:
            raise ValueError(f"Campo obrigatório ausente: {campo}")

    contexto = body["scenarioContext"]
    for campo in ["specialty", "protocol", "kbVersion"]:
        if campo not in contexto:
            raise ValueError(f"scenarioContext.{campo} é obrigatório")

    if len(body["userText"]) > MAX_USER_TEXT_CHARS:
        raise ValueError("userText excede limite configurado")

    if len(body["conversationSummary"]) > MAX_SUMMARY_CHARS:
        raise ValueError("conversationSummary excede limite configurado")

    if len(body.get("lastTurns", [])) > MAX_LAST_TURNS:
        raise ValueError("Quantidade de lastTurns excede limite")

# ============================================================
# RAG – Estruturas e funções
# ============================================================

@dataclass
class Evidencia:
    docId: str
    chunkId: str
    score: float
    snippet: str

def construir_rag_query(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constrói uma consulta de recuperação semântica (RAG).
    NÃO é SQL.
    """
    contexto = body["scenarioContext"]
    user_text = normalizar_texto(body["userText"], MAX_USER_TEXT_CHARS)
    resumo = normalizar_texto(body["conversationSummary"], MAX_SUMMARY_CHARS)

    retrieval_cfg = body.get("retrievalContext", {})
    topk = int(retrieval_cfg.get("topK", DEFAULT_TOPK))
    threshold = float(retrieval_cfg.get("scoreThreshold", DEFAULT_SCORE_THRESHOLD))

    query_text = f"{contexto['specialty']} {contexto['protocol']} {user_text} {resumo}"
    query_text = normalizar_texto(query_text, 2500)

    return {
        "queryText": query_text,
        "filters": {
            "specialty": contexto["specialty"],
            "protocol": contexto["protocol"],
            "kbVersion": contexto["kbVersion"],
        },
        "topK": min(topk, 20),
        "scoreThreshold": threshold,
    }

def recuperar_evidencias_s3_vector(rag_query: Dict[str, Any]) -> List[Evidencia]:
    """
    Placeholder para recuperação no Amazon S3 Vector.

    Em produção:
    - gerar embedding do queryText
    - buscar vetores no S3 Vector
    - aplicar filtros de metadados
    """
    return []

def avaliar_suficiencia_evidencias(
    evidencias: List[Evidencia],
    threshold: float
) -> Tuple[bool, float]:
    """
    Determina se há evidência suficiente para resposta fundamentada.
    """
    if not evidencias:
        return True, 0.0
    max_score = max(e.score for e in evidencias)
    return max_score < threshold, max_score

# ============================================================
# Prompt e Bedrock
# ============================================================

def compor_prompt(
    body: Dict[str, Any],
    evidencias: List[Evidencia],
    no_evidence: bool
) -> Tuple[str, str]:
    """
    Constrói o prompt final a partir de template versionado.
    """
    versao = body.get("promptVersion", "turn-default-v1")
    template = PROMPT_TEMPLATES.get(versao, PROMPT_TEMPLATES["turn-default-v1"])

    scenario_txt = json.dumps(body["scenarioContext"], ensure_ascii=False)
    resumo = normalizar_texto(body["conversationSummary"], MAX_SUMMARY_CHARS)
    last_turns = "\n".join(
        f"{t.get('role','').upper()}: {t.get('text','')}"
        for t in body.get("lastTurns", [])
    ) or "(nenhuma)"

    evid_txt = "\n".join(
        f"- [{e.docId}#{e.chunkId}] {normalizar_texto(e.snippet, MAX_SNIPPET_CHARS)}"
        for e in evidencias
    ) or "(nenhuma)"

    user_txt = normalizar_texto(body["userText"], MAX_USER_TEXT_CHARS)
    if no_evidence:
        user_txt += "\n\n[Sem evidência suficiente: priorize perguntas de esclarecimento]"

    prompt = template.format(
        scenario=scenario_txt,
        summary=resumo,
        last_turns=last_turns,
        evidence=evid_txt,
        user=user_txt,
    )

    return prompt, versao

def chamar_bedrock(prompt: str, generation: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Executa uma única inferência no Amazon Bedrock.
    """
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": int(generation.get("maxOutputTokens", MAX_OUTPUT_TOKENS)),
        "temperature": float(generation.get("temperature", TEMPERATURE)),
        "top_p": float(generation.get("topP", TOP_P)),
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ],
    }

    resp = bedrock.invoke_model(
        modelId=generation.get("modelId", MODEL_ID_TURN),
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )

    payload = json.loads(resp["body"].read())
    texto = payload.get("content", [{}])[0].get("text", "")
    return texto.strip(), payload.get("usage", {})

# ============================================================
# Endpoint principal
# ============================================================

@app.post("/v1/ai/turn")
def api_turn():
    inicio = time.time()

    try:
        correlation_id = exigir_header("x-correlation-id")
        session_id = exigir_header("x-session-id")
        turn_id = exigir_header("x-turn-id")
        idempotency_key = exigir_header("x-idempotency-key")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    body = request.get_json(silent=True) or {}
    try:
        validar_payload(body)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    rag_query = construir_rag_query(body)
    evidencias = recuperar_evidencias_s3_vector(rag_query)
    no_evidence, _ = avaliar_suficiencia_evidencias(
        evidencias,
        rag_query["scoreThreshold"]
    )

    prompt, prompt_version = compor_prompt(body, evidencias, no_evidence)
    texto_modelo, usage = chamar_bedrock(prompt, body.get("generationConfig", {}))

    resposta = normalizar_texto(texto_modelo, 2000)
    flags = ["no_evidence"] if no_evidence else []

    return jsonify({
        "turnId": turn_id,
        "replyText": resposta,
        "citations": [
            {"docId": e.docId, "chunkId": e.chunkId, "score": e.score}
            for e in evidencias
        ],
        "safetyFlags": flags,
        "telemetry": {
            "correlationId": correlation_id,
            "sessionId": session_id,
            "turnId": turn_id,
            "idempotencyKey": idempotency_key,
            "promptVersion": prompt_version,
            "kbVersion": body["scenarioContext"]["kbVersion"],
            "usage": usage,
            "totalLatencyMs": int((time.time() - inicio) * 1000),
        }
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
