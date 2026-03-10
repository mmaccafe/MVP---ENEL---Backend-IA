# endpoints/summarize_session_endpoint.py
from flask import Blueprint, request, jsonify, current_app

# ajustar para o módulo utilitário que você realmente tem
from utils.prompt_utils import compose_prompt
from utils.prompt_repository import PromptRepository
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

    # usar a função de composição de prompt disponível no repositório
    # (ajuste os nomes de argumentos se compose_prompt tiver assinatura diferente)
    prompt_str = compose_prompt(payload_json=payload, prompt_template_json=template)

    model_id = current_app.config.get("BEDROCK_SUMMARIZE_MODEL_ID")
    gen_cfg = payload.get("generationConfig", {})
    generation = {
        "maxOutputTokens": int(gen_cfg.get("maxOutputTokens", 400)),
        "temperature": float(gen_cfg.get("temperature", 0.2)),
        "topP": float(gen_cfg.get("topP", 0.9)),
    }

    llm = _rt.invoke_text_model(model_id=model_id, prompt=prompt_str, generation=generation)

    return jsonify({
        "newSummaryRaw": llm.get("raw"),
        "model": llm.get("modelId", model_id)
    }), 200