# =============================================================================
# Arquivo: endpoints/turn_endpoint.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Endpoint:
#   POST /v1/ai/turn
# Finalidade:
#   Processar um Turn síncrono:
#     - Resolve blueprint + componentes de templates (em português, no bundle)
#     - Executa RAG via Bedrock KB (S3 Vectors)
#     - Compoe prompt final
#     - Invoca modelo no Bedrock Runtime
#     - Retorna replyText + citations + flags + telemetria mínima
# =============================================================================

from flask import Blueprint, request, jsonify, current_app

from utils.validation_utils import validate_turn_payload
from utils.prompt_repository import PromptRepository
from utils.prompt_composer import compose_turn_prompt

from services.bedrock_kb_service import BedrockKnowledgeBaseService
from services.bedrock_runtime_service import BedrockRuntimeService

turn_bp = Blueprint("turn", __name__)

_repo = None
_kb = None
_rt = None


def _init():
    """
    Inicializa dependências compartilhadas (MVP).
    """
    global _repo, _kb, _rt

    if _repo is None:
        _repo = PromptRepository(current_app.config["TEMPLATES_ROOT"])

    if _kb is None:
        _kb = BedrockKnowledgeBaseService(
            region=current_app.config["AWS_REGION"],
            timeout_seconds=current_app.config["BEDROCK_TIMEOUT_SECONDS"],
        )

    if _rt is None:
        _rt = BedrockRuntimeService(
            region=current_app.config["AWS_REGION"],
            timeout_seconds=current_app.config["BEDROCK_TIMEOUT_SECONDS"],
        )


@turn_bp.route("/turn", methods=["POST"])
def post_turn():
    _init()

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON inválido"}), 400

    err = validate_turn_payload(payload)
    if err:
        return jsonify({"error": err}), 400

    # 1) Carrega blueprint (receita)
    pr = payload["promptRef"]
    blueprint_id = pr["blueprintId"]
    blueprint_version = pr["blueprintVersion"]

    try:
        blueprint = _repo.load_blueprint(blueprint_id, blueprint_version)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # 2) Resolve componentes do blueprint
    refs = blueprint["componentRefs"]  # ex.: {"persona": "...json", "especialidade": "...json", ...}
    try:
        components = {
            "persona": _repo.load_component("personas", refs["persona"]),
            "especialidade": _repo.load_component("especialidades", refs["especialidade"]),
            "cenario": _repo.load_component("cenarios", refs["cenario"]),
            "politicas": _repo.load_component("politicas", refs["politicas"]),
            "saida": _repo.load_component("saida", refs["saida"]),
        }
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    # 3) RAG via Knowledge Base (S3 Vectors)
    kb_id = current_app.config["BEDROCK_KB_ID"]
    if not kb_id:
        return jsonify({"error": "BEDROCK_KB_ID não configurado"}), 500

    retrieval_cfg = payload.get("retrievalConfig", {})
    top_k = int(retrieval_cfg.get("topK", current_app.config["DEFAULT_TOP_K"]))
    threshold = float(retrieval_cfg.get("scoreThreshold", current_app.config["DEFAULT_SCORE_THRESHOLD"]))
    filters = retrieval_cfg.get("filters")

    kb_resp = _kb.retrieve(
        knowledge_base_id=kb_id,
        query_text=payload["userText"],
        top_k=top_k,
        filters=filters,
    )
    rag_norm = _kb.normalize(kb_resp, score_threshold=threshold)

    # 4) Composição do prompt final
    prompt_str = compose_turn_prompt(
        payload_json=payload,
        blueprint_json=blueprint,
        components_json=components,
        rag_json=rag_norm,
    )

    # 5) Invoke Bedrock Runtime
    gen_cfg = payload.get("generationConfig", {})
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", current_app.config["DEFAULT_MAX_OUTPUT_TOKENS"])),
        "temperature": float(gen_cfg.get("temperature", current_app.config["DEFAULT_TEMPERATURE"])),
        "topP": float(gen_cfg.get("topP", current_app.config["DEFAULT_TOP_P"])),
    }

    model_id = current_app.config["BEDROCK_TURN_MODEL_ID"]
    llm = _rt.invoke_text_model(model_id=model_id, prompt=prompt_str, generation=generation)

    # 6) Retorno ao Backend App (IA não persiste)
    return jsonify({
        "replyText": llm.get("replyText", ""),
        "citations": rag_norm.get("citations", []),
        "flags": rag_norm.get("flags", []),
        "model": llm.get("modelId", model_id),
        "telemetry": {
            "blueprintId": blueprint_id,
            "blueprintVersion": blueprint_version,
            "ragTopK": top_k,
            "ragThreshold": threshold
        }
    }), 200
