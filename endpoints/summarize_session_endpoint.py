# endpoints/summarize_session_endpoint.py
from flask import Blueprint, request, jsonify
from utils.prompt_utils import compose_prompt
from utils.ia_services import (
    load_prompt_template_for_summarize,
    call_bedrock_llm_summarize,
)

summarize_bp = Blueprint("summarize", __name__)


@summarize_bp.route("/summarize-session", methods=["POST"])
def handle_summarize_session():
    """
    Endpoint de sumarização de sessão.

    Responsabilidades:
    - Receber resumo atual + últimos turns.
    - Carregar template de sumarização.
    - Compor prompt com histórico condensado.
    - Chamar LLM para gerar novo summary.
    """
    
# =============================================================================
# Arquivo: endpoints/summarize_session_endpoint.py
# Projeto: Backend de IA (MVP) - EMS GenAI
# Endpoint:
#   POST /v1/ai/summarize-session
# Finalidade:
#   Gerar/atualizar resumo de sessão (conversationSummary) após fechamento do atendimento.
#
#     Responsabilidades:
#     - Receber resumo atual + últimos turns.
#     - Carregar template de sumarização.
#     - Compor prompt com histórico condensado.
#     - Chamar LLM para gerar novo summary.


# =============================================================================

from flask import Blueprint, request, jsonify, current_app
from utils.prompt_repository import PromptRepository
from utils.prompt_composer import compose_simple_prompt
from services.bedrock_runtime_service import BedrockRuntimeService

summarize_bp = Blueprint("summarize", __name__)

_repo = None
_rt = None


def _init():
    global _repo, _rt
    if _repo is None:
        _repo = PromptRepository(current_app.config["TEMPLATES_ROOT"])
    if _rt is None:
        _rt = BedrockRuntimeService(
            region=current_app.config["AWS_REGION"],
            timeout_seconds=current_app.config["BEDROCK_TIMEOUT_SECONDS"],
        )


@summarize_bp.route("/summarize-session", methods=["POST"])
def post_summarize_session():
    _init()
    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Payload JSON inválido"}), 400

    for k in ["sessionId", "currentSummary", "recentTurns"]:
        if k not in payload:
            return jsonify({"error": f"Campo obrigatório ausente: {k}"}), 400

    template_file = payload.get("templateFile", "resumo_sessao_v1.json")
    try:
        template = _repo.load_summarize_template(template_file)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404

    prompt_str = compose_simple_prompt(payload_json=payload, template_json=template)

    model_id = current_app.config["BEDROCK_SUMMARIZE_MODEL_ID"]
    gen_cfg = payload.get("generationConfig", {})
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", 400)),
        "temperature": float(gen_cfg.get("temperature", 0.2)),
        "topP": float(gen_cfg.get("topP", 0.9)),
    }

    llm = _rt.invoke_text_model(model_id=model_id, prompt=prompt_str, generation=generation)

    # Retorna texto cru (no MVP o App Backend persiste)
    return jsonify({
        "newSummaryRaw": llm.get("raw"),
        "model": llm.get("modelId", model_id)
    }), 200
