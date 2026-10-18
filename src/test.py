import pytest
import warnings
from src.model import load_kb, query_ai_model, validate_ai_response, get_system_prompt_schema, golden_master_hash

# Force Pydantic v2 deprecation warnings (.dict, .parse_obj, .schema) to fail pytest
warnings.filterwarnings("error", category=DeprecationWarning)


def test_kb_loads_all_ten_facts():
    kb = load_kb()
    assert len(kb) == 10
    assert kb["KB-01"] == "The capital of France is Paris."
    assert kb["KB-07"] == "The chemical symbol for gold is Au."


def test_ai_model_queries_knowledge_base():
    res = query_ai_model("What is the currency of Japan?")
    assert res["structured"]["query"] == "What is the currency of Japan?"
    assert "yen" in res["structured"]["answer"].lower()
    assert res["structured"]["sources"][0]["kb_id"] == "KB-09"


def test_invalid_kb_id_regex_fails():
    bad_payload = {
        "query": "What is the capital of France?",
        "answer": "Paris",
        "confidence": 0.9,
        "sources": [{"kb_id": "BAD-ID", "fact": "The capital of France is Paris."}],
    }
    with pytest.raises(ValueError):
        validate_ai_response(bad_payload)


def test_root_validator_blocks_low_confidence():
    bad_payload = {
        "query": "When was the United Nations founded?",
        "answer": "1945",
        "confidence": 0.35,
        "sources": [{"kb_id": "KB-08", "fact": "The United Nations was founded in 1945."}],
        "verified": True,
    }
    with pytest.raises(ValueError, match="Low-confidence"):
        validate_ai_response(bad_payload)


def test_schema_export_and_golden_hash():
    schema = get_system_prompt_schema()
    assert schema["title"] == "AIQueryResult"
    assert schema["required"] == ["answer", "confidence", "query", "sources"]
    assert len(golden_master_hash()) == 64


def test_validation_speed_benchmark():
    kb = load_kb()
    sample = {
        "query": "  How many chambers does the human heart have?  ",
        "answer": "  The human heart has four chambers: two atria and two ventricles.  ",
        "confidence": 0.98,
        "sources": [{"kb_id": "KB-04", "fact": "The human heart has four chambers: two atria and two ventricles."}],
        "verified": True,
    }
    for _ in range(1500):
        validate_ai_response(sample, kb)
