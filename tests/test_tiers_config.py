from app.core.config import settings


def test_load_tiers_from_yaml():
    """Verifies that model tiers are loaded from config/tiers.yaml in order."""
    tiers = settings.tiers

    assert len(tiers) >= 2
    assert tiers[0].name == "small"
    assert tiers[0].model == "ollama_chat/llama3.2:3b"
    assert tiers[0].price_per_1m == 0.0
    assert tiers[0].shadow_price_per_1m.input == 0.15
    assert tiers[0].shadow_price_per_1m.output == 0.60
    assert tiers[0].input_price == 0.15
    assert tiers[0].output_price == 0.60

    assert tiers[1].name == "large"
    assert tiers[1].model == "ollama_chat/llama3.1:8b"
    assert tiers[1].price_per_1m == 0.0
    assert tiers[1].shadow_price_per_1m.input == 2.00
    assert tiers[1].shadow_price_per_1m.output == 8.00
    assert tiers[1].input_price == 2.00
    assert tiers[1].output_price == 8.00


def test_tiers_ordered_cheapest_first():
    """Verifies that the N tiers are sorted from lowest to highest shadow price."""
    tiers = settings.tiers
    shadow_prices = [t.input_price for t in tiers]
    assert shadow_prices == sorted(shadow_prices)


def test_get_tier_helper():
    """Verifies retrieval of tier configuration by name."""
    small = settings.get_tier("small")
    large = settings.get_tier("large")
    unknown = settings.get_tier("nonexistent")

    assert small.name == "small"
    assert large.name == "large"
    # Unknown tier defaults safely to first (lowest) tier
    assert unknown.name == "small"
