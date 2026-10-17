import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI
from schema import AIQueryResult

KB_FILE = Path(__file__).parent / "knowledge_base.txt"


def load_kb() -> Dict[str, str]:
    """Reads non-empty lines from knowledge_base.txt into {'KB-01': '...', ..., 'KB-10': '...'}."""
    lines = [line.strip() for line in KB_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {f"KB-{idx:02d}": text for idx, text in enumerate(lines, start=1)}


def get_system_prompt_schema() -> Dict[str, Any]:
    """Generates the JSON schema injected into the AI system prompt."""
    # Breaks in v2: .schema() -> use .model_json_schema()
    s = AIQueryResult.schema()
    return {"title": s["title"], "required": sorted(s.get("required", [])), "properties": list(s.get("properties", {}).keys())}


def query_ai_model(question: str) -> Dict[str, Any]:
    """
    Uses an AI model on Nebius Token Factory to answer a question from knowledge_base.txt
    following the strict Pydantic schema from schema.py.
    """
    kb = load_kb()
    schema_def = get_system_prompt_schema()
    api_key = os.environ.get("NEBIUS_API_KEY")

    if api_key:
        client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=api_key,
        )
        kb_context = "\n".join(f"{k}: {v}" for k, v in kb.items())
        res = client.chat.completions.create(
            model="nvidia/Nemotron-3.5-Lightning",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Answer using ONLY the knowledge base below. "
                        f"Return strict JSON matching this schema: {json.dumps(schema_def)}\n\n"
                        f"Knowledge Base:\n{kb_context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_payload = json.loads(res.choices[0].message.content)
    else:
        # Deterministic fallback for fast offline unit testing & benchmarking
        keywords = [w.lower() for w in question.replace("?", "").split() if len(w) > 3]
        raw_payload = {"query": question, "answer": "Not found", "confidence": 0.2, "sources": [], "verified": False}
        for kb_id, text in kb.items():
            if any(k in text.lower() for k in keywords):
                raw_payload = {
                    "query": f"  {question}  ",
                    "answer": f"  {text}  ",
                    "confidence": 0.95,
                    "sources": [{"kb_id": kb_id, "fact": text}],
                    "verified": True,
                }
                break

    return validate_ai_response(raw_payload, kb)


def validate_ai_response(raw_payload: Dict[str, Any], kb: Dict[str, str] = None) -> Dict[str, Any]:
    """Validates raw AI JSON output against AIQueryResult and serializes it."""
    kb = kb or load_kb()
    # Breaks in v2: .parse_obj() -> use .model_validate()
    validated = AIQueryResult.parse_obj(raw_payload)

    for src in validated.sources:
        if src.kb_id not in kb:
            raise ValueError(f"Unknown KB ID: {src.kb_id}")

    # Breaks in v2: .dict() & .json() -> use .model_dump() & .model_dump_json()
    return {
        "structured": validated.dict(),
        "serialized": validated.json(sort_keys=True),
    }


def golden_master_hash() -> str:
    """Produces deterministic SHA-256 fingerprint for Stage 1 vs Stage 5 parity check."""
    kb = load_kb()
    sample = {
        "query": "  What is the highest mountain above sea level?  ",
        "answer": "  Mount Everest is the highest mountain above sea level.  ",
        "confidence": 0.95,
        "sources": [{"kb_id": "KB-06", "fact": "Mount Everest is the highest mountain above sea level."}],
        "verified": True,
    }
    out = validate_ai_response(sample, kb)
    blob = json.dumps({"out": out["structured"], "schema": get_system_prompt_schema()}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()