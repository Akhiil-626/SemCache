import pytest
from app.core.config import ModelTierConfig, ShadowPriceConfig
from app.routing.router import CostAwareRouter


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
