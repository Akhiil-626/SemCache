import pytest
from app.core.config import ModelTierConfig, ShadowPriceConfig
from app.routing.router import CostAwareRouter, RuleRouter


@pytest.fixture
def custom_tiers():
    return [
        ModelTierConfig(
            name="small",
            model="ollama_chat/llama3.2:3b",
            price_per_1m=0.0,
            shadow_price_per_1m=ShadowPriceConfig(input=0.15, output=0.60),
        ),
        ModelTierConfig(
            name="large",
            model="ollama_chat/llama3.1:8b",
            price_per_1m=0.0,
            shadow_price_per_1m=ShadowPriceConfig(input=2.50, output=10.00),
        ),
    ]


def test_routing_simple_query(custom_tiers):
    router = CostAwareRouter(tiers=custom_tiers)
    decision = router.route("What is the capital of France?")
    assert decision.tier == "small"
    assert decision.selected_model == "ollama_chat/llama3.2:3b"


def test_routing_complex_query(custom_tiers):
    router = CostAwareRouter(tiers=custom_tiers)
    decision = router.route(
        "Analyze and architect a distributed semantic cache system step by step."
    )
    assert decision.tier == "large"
    assert decision.selected_model == "ollama_chat/llama3.1:8b"


def test_routing_budget_constraint_downgrade(custom_tiers):
    router = CostAwareRouter(tiers=custom_tiers)
    # Extremely small budget forces downgrade from large to small tier
    decision = router.route(
        "Architect and optimize an algorithmic proof step by step.",
        user_budget=0.000001,
    )
    assert decision.tier == "small"
    assert "downgraded" in decision.rationale.lower()


def test_rule_positive_and_negative_firing():
    """Verifies that every rule and concept group fires on a positive example and not on a negative one."""
    router = RuleRouter()

    # 1. long_prompt (word_count > 60)
    pos_long = "word " * 65
    neg_long = "word " * 20
    _, hits = router.evaluate_features(pos_long)
    assert any("long prompt" in h for h in hits)
    _, hits = router.evaluate_features(neg_long)
    assert not any("long prompt" in h for h in hits)

    # 2. very_long_prompt (word_count > 200)
    pos_very_long = "word " * 205
    neg_very_long = "word " * 100
    _, hits = router.evaluate_features(pos_very_long)
    assert any("very long prompt" in h for h in hits)
    _, hits = router.evaluate_features(neg_very_long)
    assert not any("very long prompt" in h for h in hits)

    # 3. code_fence (```)
    pos_fence = "Here is code:\n```python\nprint(1)\n```"
    neg_fence = "Here is regular text without markdown fences."
    _, hits = router.evaluate_features(pos_fence)
    assert any("code fence" in h for h in hits)
    _, hits = router.evaluate_features(neg_fence)
    assert not any("code fence" in h for h in hits)

    # 4. code_markers group (def, class, traceback, etc.)
    pos_marker = "def calculate_average(items): return sum(items)"
    neg_marker = "The quick brown fox jumps over the lazy dog."
    _, hits = router.evaluate_features(pos_marker)
    assert any("group 'code_markers'" in h for h in hits)
    _, hits = router.evaluate_features(neg_marker)
    assert not any("group 'code_markers'" in h for h in hits)

    # 5. math_operators (digits with operators)
    pos_math = "Calculate 15 * 4 + 10 = 70 and verify 75% accuracy."
    neg_math = "No mathematical operations in this sentence."
    _, hits = router.evaluate_features(pos_math)
    assert any("math expression" in h for h in hits)
    _, hits = router.evaluate_features(neg_math)
    assert not any("math expression" in h for h in hits)

    # 6. math_logic group (calculate, probability, induction, etc.)
    pos_math_kw = "Please calculate the total posterior probability."
    neg_math_kw = "Tell me a short bedtime story."
    _, hits = router.evaluate_features(pos_math_kw)
    assert any("group 'math_logic'" in h for h in hits)
    _, hits = router.evaluate_features(neg_math_kw)
    assert not any("group 'math_logic'" in h for h in hits)

    # 7. reasoning groups (analysis, trade_offs)
    pos_reasoning = "Analyze the trade-offs between SQL and NoSQL."
    neg_reasoning = "What time does the library open today?"
    _, hits = router.evaluate_features(pos_reasoning)
    assert any("group 'analysis'" in h for h in hits)
    assert any("group 'trade_offs'" in h for h in hits)
    _, hits = router.evaluate_features(neg_reasoning)
    assert not any("group 'analysis'" in h for h in hits)

    # 8. format_requests group (json, table, schema, etc.)
    pos_format = "Provide the output strictly in valid JSON format."
    neg_format = "Write a creative narrative paragraph."
    _, hits = router.evaluate_features(pos_format)
    assert any("group 'format_requests'" in h for h in hits)
    _, hits = router.evaluate_features(neg_format)
    assert not any("group 'format_requests'" in h for h in hits)

    # 9. subtasks (multiple questions or numbered list items)
    pos_subtask = "1) Do the first step. 2) Do the second step."
    neg_subtask = "This is a single uninterrupted instruction."
    _, hits = router.evaluate_features(pos_subtask)
    assert any("multiple subtasks" in h for h in hits)
    _, hits = router.evaluate_features(neg_subtask)
    assert not any("multiple subtasks" in h for h in hits)

    # 10. message_count (multi-turn conversation >= 3)
    pos_msgs = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "how are you?"},
    ]
    neg_msgs = [{"role": "user", "content": "hello"}]
    _, hits = router.evaluate_features("how are you?", messages=pos_msgs)
    assert any("multi-turn conversation" in h for h in hits)
    _, hits = router.evaluate_features("hello", messages=neg_msgs)
    assert not any("multi-turn conversation" in h for h in hits)

    # 11. non_ascii_ratio
    pos_non_ascii = "こんにちは世界。これは並行処理のテストです。"
    neg_non_ascii = "Hello world, standard English text only."
    _, hits = router.evaluate_features(pos_non_ascii)
    assert any("non-ascii prompt" in h for h in hits)
    _, hits = router.evaluate_features(neg_non_ascii)
    assert not any("non-ascii prompt" in h for h in hits)


def test_keyword_word_boundary_matching():
    """Verifies that keyword matching strictly respects word boundaries ('complexity' != 'complex')."""
    router = RuleRouter()

    # 'complexity' must NOT match keyword 'complex'
    decision_complexity = router.route("The computational complexity of this indexing method is high.")
    assert not any("complex_reasoning" in h for h in decision_complexity.rule_hits)

    # 'complex' as standalone word MUST match keyword 'complex'
    decision_complex = router.route("This is a complex system architecture.")
    assert any("group 'complex_reasoning'" in h for h in decision_complex.rule_hits)


def test_threshold_score_mapping(custom_tiers):
    """Verifies score mapping to small tier below threshold and large tier at or above threshold."""
    router = RuleRouter(tiers=custom_tiers)

    # score 0.0 < 2.0 -> small
    decision_low = router.route("What is photosynthesis?")
    assert decision_low.score < router.large_threshold
    assert decision_low.tier == "small"

    # score 6.0 >= 2.0 -> large (analysis +2, comparison +2, trade_offs +2)
    decision_high = router.route("Analyze and compare the trade-offs between two database engines.")
    assert decision_high.score >= router.large_threshold
    assert decision_high.tier == "large"


def test_explicit_model_bypasses_scoring(custom_tiers):
    """Verifies that explicit model selection bypasses feature scoring completely."""
    router = RuleRouter(tiers=custom_tiers)

    # Even on a highly complex query, explicit 'small' bypasses scoring
    decision_small = router.route(
        "Analyze and architect a distributed system step by step with proof.",
        model="small",
    )
    assert decision_small.tier == "small"
    assert decision_small.desired_tier == "small"
    assert decision_small.score == 0.0
    assert decision_small.confidence == 1.0
    assert decision_small.rule_hits == []
    assert "bypasses scoring" in decision_small.reason

    # On a simple query, explicit 'large' bypasses scoring
    decision_large = router.route("Hello world", model="large")
    assert decision_large.tier == "large"
    assert decision_large.desired_tier == "large"
    assert decision_large.score == 0.0
    assert decision_large.confidence == 1.0
    assert decision_large.rule_hits == []
    assert "bypasses scoring" in decision_large.reason


def test_rationale_lists_exact_rules_fired(custom_tiers):
    """Verifies that routing rationale lists exactly the rules that actually fired."""
    router = RuleRouter(tiers=custom_tiers)

    # Prompt triggering code fence (+2), group 'proof' (+2), long prompt (+1), and math expression (+1.5)
    prompt = "```\nprove that 1=1\n```\n" + " word" * 65
    decision = router.route(prompt)

    assert "code fence (+2)" in decision.reason
    assert "group 'proof' (+2)" in decision.reason
    assert "long prompt (+1)" in decision.reason
    assert "math expression (+1.5)" in decision.reason
    # Expected 4 rule hits: code fence, proof group, long prompt, and math expression (legitimately fired on '1=1')
    assert len(decision.rule_hits) == 4
    for hit in decision.rule_hits:
        assert hit in decision.reason


def test_yaml_config_changes_router_behavior(custom_tiers):
    """Verifies that router loaded from modified YAML configuration changes routing behavior."""
    router = RuleRouter(tiers=custom_tiers)
    prompt = "Analyze and architect a cache system."

    # Under default threshold (2.0), score is 4.0 -> tier is large
    decision_before = router.route(prompt)
    assert decision_before.tier == "large"

    # Now reload with a higher threshold of 10.0
    router.load_config({
        "thresholds": {"large_threshold": 10.0},
        "tiers": {"default_tier": "small", "large_tier": "large"},
        "concept_groups": router.concept_groups,
        "feature_rules": router.feature_rules,
    })

    # Under threshold 10.0, score 4.0 < 10.0 -> tier is now small
    decision_after = router.route(prompt)
    assert decision_after.tier == "small"
    assert decision_after.score == decision_before.score


def test_repeated_word_counts_once(custom_tiers):
    """PART 6: Verifies that a repeated keyword fires its group at most once per request."""
    router = RuleRouter(tiers=custom_tiers)
    prompt = "Please analyze and analyze and analyze this problem thoroughly."
    decision = router.route(prompt)

    analysis_hits = [h for h in decision.rule_hits if "group 'analysis'" in h]
    assert len(analysis_hits) == 1
    assert decision.score == 2.0


def test_variants_in_one_group_count_once(custom_tiers):
    """PART 6: Verifies that multiple variants within the same concept group count at most once."""
    router = RuleRouter(tiers=custom_tiers)
    prompt = "We must prove this theorem with a rigorous proof while proving each lemma."
    decision = router.route(prompt)

    proof_hits = [h for h in decision.rule_hits if "group 'proof'" in h]
    assert len(proof_hits) == 1
    assert decision.score == 2.0


def test_dates_and_ranges_do_not_trigger_math_rule(custom_tiers):
    """PART 6: Verifies that dates like 2026-10-13 and ranges like 3-5 do not trigger the math rule."""
    router = RuleRouter(tiers=custom_tiers)

    # Text containing dates, ranges, and version numbers without genuine math
    text_non_math = "Meeting scheduled on 2026-10-13 to review pages 3-5 of draft v1.0."
    decision_non_math = router.route(text_non_math)
    assert not any("math expression" in h for h in decision_non_math.rule_hits)

    # Genuine mathematical equation with operator or equals sign
    text_genuine_math = "Calculate 2 + 2 = 4 and verify that 1=1."
    decision_math = router.route(text_genuine_math)
    assert any("math expression" in h for h in decision_math.rule_hits)


def test_concept_group_config_loads_from_yaml():
    """PART 6: Verifies that concept groups are correctly loaded from YAML configuration."""
    router = RuleRouter()
    assert len(router.concept_groups) > 0

    group_names = [g["name"] for g in router.concept_groups]
    assert "proof" in group_names
    assert "architecture" in group_names
    assert "step_by_step" in group_names
    assert "analysis" in group_names

    proof_group = next(g for g in router.concept_groups if g["name"] == "proof")
    assert "prove" in proof_group["words"]
    assert "proof" in proof_group["words"]
    assert proof_group["weight"] == 2.0


def test_config_lint_no_duplicate_keywords():
    """PART 0: Config-lint test that fails if any word appears in two groups or two lists in config/routing.yaml."""
    from pathlib import Path
    import yaml
    from app.core.config import settings

    config_path = Path(settings.ROUTING_CONFIG_PATH)
    assert config_path.exists(), f"Routing config not found at {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    seen_words: dict[str, str] = {}
    duplicates: list[str] = []

    # Check concept groups
    for group in data.get("concept_groups", []):
        g_name = group.get("name", "unnamed")
        for word in group.get("words", []):
            cleaned = word.strip().lower()
            if cleaned in seen_words:
                duplicates.append(f"'{cleaned}' in group '{g_name}' (already in '{seen_words[cleaned]}')")
            else:
                seen_words[cleaned] = f"group:{g_name}"

    # Check any standalone keyword lists if configured
    for list_key in ("reasoning_keywords", "code_markers", "math_markers", "format_requests"):
        if list_key in data:
            for word in data[list_key]:
                cleaned = word.strip().lower()
                if cleaned in seen_words:
                    duplicates.append(f"'{cleaned}' in list '{list_key}' (already in '{seen_words[cleaned]}')")
                else:
                    seen_words[cleaned] = f"list:{list_key}"

    assert not duplicates, f"Duplicate keywords found across groups/lists: {duplicates}"

