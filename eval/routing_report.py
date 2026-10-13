"""Evaluation report for cost-aware tier router.

Evaluates routing rules across benchmark prompts in eval/routing_prompts.json
without invoking external LLMs, networks, or databases.
"""
import argparse
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure clean UTF-8 console output in Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.routing.router import RuleRouter


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tier router against benchmark prompts.")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to prompts JSON file (default: eval/routing_prompts.json)",
    )
    args = parser.parse_args()

    if args.file:
        prompts_path = Path(args.file)
    else:
        prompts_path = Path(__file__).resolve().parent / "routing_prompts.json"
        if not prompts_path.exists():
            prompts_path = PROJECT_ROOT / "eval" / "routing_prompts.json"

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    router = RuleRouter()

    results = []
    for item in prompts:
        decision = router.route(item["prompt"])
        results.append({
            "id": item["id"],
            "difficulty": item["difficulty"],
            "expected_tier": item["expected_tier"],
            "chosen_tier": decision.tier,
            "score": decision.score,
            "rules_fired": decision.rule_hits,
        })

    # 1. Per-Prompt Table
    header = f"{'ID':>3}  {'Difficulty':<10} {'Expected':<8} {'Chosen':<8} {'Score':>5}  {'Rules Fired'}"
    divider = "=" * 100
    print("\n" + divider)
    print("ROUTER BENCHMARK EVALUATION REPORT")
    print(divider)
    print(header)
    print("-" * 100)

    for r in results:
        rules_str = ", ".join(r["rules_fired"]) if r["rules_fired"] else "-"
        print(
            f"{r['id']:>3}  {r['difficulty']:<10} {r['expected_tier']:<8} "
            f"{r['chosen_tier']:<8} {r['score']:>5.1f}  {rules_str}"
        )
    print(divider)

    # 2. Accuracy
    easy_total = sum(1 for r in results if r["difficulty"] == "easy")
    easy_correct = sum(1 for r in results if r["difficulty"] == "easy" and r["chosen_tier"] == "small")
    hard_total = sum(1 for r in results if r["difficulty"] == "hard")
    hard_correct = sum(1 for r in results if r["difficulty"] == "hard" and r["chosen_tier"] == "large")

    easy_acc = (easy_correct / easy_total * 100.0) if easy_total else 0.0
    hard_acc = (hard_correct / hard_total * 100.0) if hard_total else 0.0
    overall_acc = ((easy_correct + hard_correct) / (easy_total + hard_total) * 100.0) if (easy_total + hard_total) else 0.0

    print("\n--- Accuracy Metrics ---")
    print(f"Easy prompts (expected 'small') : {easy_correct}/{easy_total} ({easy_acc:.1f}%)")
    print(f"Hard prompts (expected 'large') : {hard_correct}/{hard_total} ({hard_acc:.1f}%)")
    print(f"Overall target accuracy        : {easy_correct + hard_correct}/{easy_total + hard_total} ({overall_acc:.1f}%)")

    # 3. Medium Distribution
    med_total = sum(1 for r in results if r["difficulty"] == "medium")
    med_small = sum(1 for r in results if r["difficulty"] == "medium" and r["chosen_tier"] == "small")
    med_large = sum(1 for r in results if r["difficulty"] == "medium" and r["chosen_tier"] == "large")

    med_small_pct = (med_small / med_total * 100.0) if med_total else 0.0
    med_large_pct = (med_large / med_total * 100.0) if med_total else 0.0

    print("\n--- Medium Distribution (expected 'either') ---")
    print(f"Total medium prompts           : {med_total}")
    print(f"Routed to 'small'              : {med_small} ({med_small_pct:.1f}%)")
    print(f"Routed to 'large'              : {med_large} ({med_large_pct:.1f}%)")

    # 4. Confusion Matrix
    print("\n--- Confusion Matrix ---")
    print(f"{'Expected Category':<22} {'Chosen small':>14} {'Chosen large':>14} {'Total':>8}")
    print("-" * 62)
    print(f"{'easy (expected small)':<22} {easy_correct:>14} {easy_total - easy_correct:>14} {easy_total:>8}")
    print(f"{'hard (expected large)':<22} {hard_total - hard_correct:>14} {hard_correct:>14} {hard_total:>8}")
    print(f"{'medium (expected either)':<22} {med_small:>14} {med_large:>14} {med_total:>8}")
    print("-" * 62)
    total_small = easy_correct + (hard_total - hard_correct) + med_small
    total_large = (easy_total - easy_correct) + hard_correct + med_large
    total_all = len(results)
    print(f"{'Total':<22} {total_small:>14} {total_large:>14} {total_all:>8}")

    # 5. Threshold Sweep
    thresholds = [round(t * 0.5, 1) for t in range(2, 17)]  # 1.0 to 8.0, step 0.5
    print("\n--- Threshold Sweep (LARGE_THRESHOLD: 1.0 to 8.0, step 0.5) ---")
    sweep_header = f"{'Threshold':>9}  {'Easy->Small %':>14}  {'Hard->Large %':>14}  {'Med->Small %':>13}  {'Med->Large %':>13}"
    print(sweep_header)
    print("-" * len(sweep_header))

    for th in thresholds:
        th_easy_small = sum(1 for r in results if r["difficulty"] == "easy" and r["score"] < th)
        th_hard_large = sum(1 for r in results if r["difficulty"] == "hard" and r["score"] >= th)
        th_med_small = sum(1 for r in results if r["difficulty"] == "medium" and r["score"] < th)
        th_med_large = sum(1 for r in results if r["difficulty"] == "medium" and r["score"] >= th)

        e_pct = (th_easy_small / easy_total * 100.0) if easy_total else 0.0
        h_pct = (th_hard_large / hard_total * 100.0) if hard_total else 0.0
        ms_pct = (th_med_small / med_total * 100.0) if med_total else 0.0
        ml_pct = (th_med_large / med_total * 100.0) if med_total else 0.0

        marker = " *" if th == router.large_threshold else ""
        print(f"{th:>9.1f}  {e_pct:>13.1f}%  {h_pct:>13.1f}%  {ms_pct:>12.1f}%  {ml_pct:>12.1f}%{marker}")
    print("-" * len(sweep_header))
    print("(* denotes current configured threshold)\n")


if __name__ == "__main__":
    main()
