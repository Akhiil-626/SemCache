import pytest
from app.llm.client import LLMResponse
from app.routing.validators import (
    ValidationResult,
    run_validators,
    validate_answer,
    validate_no_repetition,
    validate_non_empty,
    validate_not_refusal,
    validate_not_truncated,
    validate_valid_json,
    validation_pipeline,
)


def _make_response(
    content: str,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    finish_reason: str = "stop",
    tier: str = "small",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=100.0,
        real_cost=0.0,
        shadow_cost=0.0001,
        model_name="test-model",
        tier=tier,
        finish_reason=finish_reason,
    )


def test_validate_non_empty_passes_and_fails():
    """Verifies that non_empty passes on valid text and fails on empty/short responses."""
    # Good input
    good_resp = _make_response("This is a valid answer.", completion_tokens=5)
    res = validate_non_empty(good_resp, {"tier": "small"})
    assert res.passed is True
    assert res.reason is None

    # Empty string
    empty_resp = _make_response("", completion_tokens=0)
    res = validate_non_empty(empty_resp, {"tier": "small"})
    assert res.passed is False
    assert res.reason == "empty_response"

    # Whitespace only
    whitespace_resp = _make_response("   \n\t  ", completion_tokens=2)
    res = validate_non_empty(whitespace_resp, {"tier": "small"})
    assert res.passed is False
    assert res.reason == "empty_response"

    # Token count below minimum
    cfg = {"min_completion_tokens": {"small": 5, "default": 1}}
    short_resp = _make_response("Hi", completion_tokens=2)
    res = validate_non_empty(short_resp, {"tier": "small"}, config=cfg)
    assert res.passed is False
    assert res.reason == "empty_response"


def test_validate_not_truncated_passes_and_fails():
    """Verifies that not_truncated passes on 'stop' and fails on 'length'."""
    # Good input
    normal_resp = _make_response("Completed normally.", finish_reason="stop")
    res = validate_not_truncated(normal_resp)
    assert res.passed is True
    assert res.reason is None

    # Truncated input
    truncated_resp = _make_response("Sentence cut off mid-way...", finish_reason="length")
    res = validate_not_truncated(truncated_resp)
    assert res.passed is False
    assert res.reason == "truncated"


def test_validate_not_refusal_passes_and_fails():
    """Verifies refusal detection on short responses (<60 tokens) and allowance for longer discussions."""
    # Short refusal
    refusal_resp = _make_response(
        "I apologize, but I cannot assist with that request.",
        completion_tokens=10,
    )
    res = validate_not_refusal(refusal_resp)
    assert res.passed is False
    assert res.reason == "refusal"

    # AI identity refusal pattern
    ai_resp = _make_response(
        "As an AI, I am unable to perform physical tasks.",
        completion_tokens=12,
    )
    res = validate_not_refusal(ai_resp)
    assert res.passed is False
    assert res.reason == "refusal"

    # Legitimate non-refusal answer
    ok_resp = _make_response(
        "To reset your router, press and hold the reset pin for 10 seconds.",
        completion_tokens=15,
    )
    res = validate_not_refusal(ok_resp)
    assert res.passed is True
    assert res.reason is None

    # Long response containing refusal words should NOT fail (only for short answers < 60 tokens)
    long_essay = (
        "In this analysis, we examine why an author might say 'I cannot fulfill' in a historical drama. "
        + "The protagonist faces moral dilemmas and personal boundaries that define character arcs. " * 5
    )
    long_resp = _make_response(long_essay, completion_tokens=85)
    res = validate_not_refusal(long_resp)
    assert res.passed is True
    assert res.reason is None


def test_validate_valid_json_passes_and_fails():
    """Verifies JSON syntax validation and optional JSON schema validation."""
    # No response_format requested -> always passes
    raw_text = _make_response("This is plain text.")
    res = validate_valid_json(raw_text, request_context={})
    assert res.passed is True

    # Valid JSON
    valid_json_resp = _make_response('{"name": "Alice", "age": 30}')
    res = validate_valid_json(
        valid_json_resp,
        request_context={"response_format": {"type": "json_object"}},
    )
    assert res.passed is True
    assert res.reason is None

    # Invalid JSON
    invalid_json_resp = _make_response('{"name": "Alice", "age": }')
    res = validate_valid_json(
        invalid_json_resp,
        request_context={"response_format": {"type": "json_object"}},
    )
    assert res.passed is False
    assert res.reason == "invalid_json"

    # Schema validation: conforming
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "number"},
        },
        "required": ["name", "age"],
    }
    schema_ctx = {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        }
    }
    res_schema_ok = validate_valid_json(valid_json_resp, request_context=schema_ctx)
    assert res_schema_ok.passed is True

    # Schema validation: mismatching type or missing required
    mismatch_resp = _make_response('{"name": "Alice", "age": "thirty"}')
    res_schema_fail = validate_valid_json(mismatch_resp, request_context=schema_ctx)
    assert res_schema_fail.passed is False
    assert res_schema_fail.reason == "schema_mismatch"


def test_validate_no_repetition_passes_and_fails():
    """Verifies that degenerative repetition loops and excessive n-gram repeats fail validation."""
    # Short responses (<200 chars) are skipped by length check
    short_rep = _make_response("echo echo echo echo")
    res_short = validate_no_repetition(short_rep)
    assert res_short.passed is True

    # Varied normal long text (>200 chars) passes
    varied_text = (
        "The quick brown fox jumps over the lazy dog in the dense autumnal forest. "
        "Sunlight filters through the deciduous canopy, illuminating a variety of mosses, "
        "lichen, and fallen foliage that carpet the nutrient-rich earthen floor."
    )
    normal_resp = _make_response(varied_text)
    res_normal = validate_no_repetition(normal_resp)
    assert res_normal.passed is True
    assert res_normal.reason is None

    # Degenerative loop with low zlib ratio (repeating phrase 30 times)
    repeating_loop = "and then and then and then " * 30
    assert len(repeating_loop) > 200
    loop_resp = _make_response(repeating_loop)
    res_loop = validate_no_repetition(loop_resp)
    assert res_loop.passed is False
    assert res_loop.reason in ("repetition_loop", "repetition_ngram")


def test_validation_pipeline_dispatches_configured_validators():
    """Verifies that validation pipeline correctly aggregates failure reasons across validators."""
    # Response that is both truncated and an empty response
    bad_resp = _make_response("", finish_reason="length", completion_tokens=0)
    passed, reasons = validation_pipeline.validate(bad_resp, tier="small")
    assert passed is False
    assert "empty_response" in reasons
    assert "truncated" in reasons
