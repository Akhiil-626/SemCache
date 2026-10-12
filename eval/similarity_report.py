import asyncio
import json
import math
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache.semantic_guard import check_semantic_guard
from app.core.config import settings
from app.embeddings.service import OllamaEmbedder


def dot_product(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity for unit-normalized vectors."""
    return sum(a * b for a, b in zip(v1, v2))


async def run_evaluation() -> None:
    pairs_file = Path(__file__).resolve().parent / "pairs.json"
    if not pairs_file.exists():
        print(f"Error: {pairs_file} not found.")
        sys.exit(1)

    with open(pairs_file, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    embedder = OllamaEmbedder()
    current_threshold = settings.SIMILARITY_THRESHOLD

    print("=" * 100)
    print(f"SEMANTIC CACHE EVALUATION REPORT (Model: {embedder.model}, Threshold: {current_threshold:.2f})")
    print("=" * 100)
    print("Embedding text pairs via Ollama...")

    evaluated: list[dict] = []
    for item in pairs:
        pair_id = item["id"]
        q = item["query"]
        c = item["cached"]
        label = item["label"]

        v_q = await embedder.get_embedding(q)
        v_c = await embedder.get_embedding(c)

        if v_q is None or v_c is None:
            print(f"ERROR: Could not get embeddings for pair {pair_id} from {embedder.base_url}")
            print("Ensure Ollama is running and model is pulled: ollama pull all-minilm")
            sys.exit(1)

        sim = max(-1.0, min(1.0, dot_product(v_q, v_c)))
        guard_ok, guard_reason = check_semantic_guard(q, c)
        would_hit = (sim >= current_threshold) and guard_ok

        evaluated.append({
            "id": pair_id,
            "query": q,
            "cached": c,
            "label": label,
            "similarity": sim,
            "guard_ok": guard_ok,
            "guard_reason": guard_reason,
            "would_hit": would_hit,
        })

    # Print Table by Group
    labels_order = ["paraphrase", "unrelated", "near_opposite", "adversarial"]
    header = f"{'ID':<4} | {'Label':<14} | {'Sim':<7} | {'Guard':<8} | {'Hit?':<6} | {'Pair (Query vs Cached)'}"

    for grp in labels_order:
        grp_items = [e for e in evaluated if e["label"] == grp]
        if not grp_items:
            continue
        print("\n" + "-" * 100)
        print(f"GROUP: {grp.upper()} ({len(grp_items)} pairs)")
        print("-" * 100)
        print(header)
        print("-" * 100)
        for item in grp_items:
            guard_str = "PASS" if item["guard_ok"] else f"REJ({item['guard_reason']})"
            hit_str = "HIT" if item["would_hit"] else "MISS"
            pair_str = f"\"{item['query']}\" vs \"{item['cached']}\""
            if len(pair_str) > 55:
                pair_str = pair_str[:52] + "..."
            print(
                f"{item['id']:<4} | {item['label']:<14} | {item['similarity']:.4f}  | {guard_str:<8} | {hit_str:<6} | {pair_str}"
            )

    print("\n" + "=" * 90)
    print("THRESHOLD SWEEP (0.80 to 0.95 in steps of 0.01)")
    print("=" * 90)
    sweep_header = f"{'Threshold':<11} | {'Paraphrase Hit Rate':<22} | {'Raw False-Hits':<22} | {'Guarded False-Hits':<22}"
    print(sweep_header)
    print("-" * 90)

    paraphrases = [e for e in evaluated if e["label"] == "paraphrase"]
    non_paraphrases = [e for e in evaluated if e["label"] in ("unrelated", "near_opposite", "adversarial")]

    total_para = len(paraphrases)
    total_non_para = len(non_paraphrases)

    # Sweep from 0.80 to 0.95 in 0.01 increments
    for i in range(80, 96):
        th = round(i / 100.0, 2)
        para_hits = sum(1 for e in paraphrases if e["similarity"] >= th and e["guard_ok"])
        raw_false_hits = sum(1 for e in non_paraphrases if e["similarity"] >= th)
        guarded_false_hits = sum(1 for e in non_paraphrases if e["similarity"] >= th and e["guard_ok"])

        para_rate = (para_hits / total_para) * 100.0 if total_para else 0.0
        raw_false_rate = (raw_false_hits / total_non_para) * 100.0 if total_non_para else 0.0
        guarded_false_rate = (guarded_false_hits / total_non_para) * 100.0 if total_non_para else 0.0

        marker = " <-- current" if abs(th - current_threshold) < 0.001 else ""
        print(
            f"{th:.2f}        | {para_hits}/{total_para} ({para_rate:5.1f}%)        | {raw_false_hits}/{total_non_para} ({raw_false_rate:5.1f}%)        | {guarded_false_hits}/{total_non_para} ({guarded_false_rate:5.1f}%){marker}"
        )

    print("=" * 90)


if __name__ == "__main__":
    asyncio.run(run_evaluation())
