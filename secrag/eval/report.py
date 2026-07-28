"""M3: aggregate per-question eval results into a metrics report."""
from collections import defaultdict


def build_report(results: list[dict]) -> str:
    """`results` items: {question_type, correct, evidence_recall, ...efficiency metrics}.
    Returns a printable report covering both correctness and efficiency (tokens/cost/time) —
    efficiency fields are optional so this stays compatible with older result dicts.
    """
    lines = []
    n = len(results)
    accuracy = sum(r["correct"] for r in results) / n if n else 0.0
    recall = sum(r["evidence_recall"] for r in results) / n if n else 0.0

    lines.append(f"Overall: {n} questions, accuracy={accuracy:.1%}, evidence_recall@k={recall:.1%}")
    lines.append(f"(FinanceBench published baselines: naive ~19%, per-doc ~50%, full-context ~79%, oracle ~85%)")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    lines.append("\nBy question type:")
    for qtype, items in sorted(by_type.items()):
        m = len(items)
        acc = sum(r["correct"] for r in items) / m
        rec = sum(r["evidence_recall"] for r in items) / m
        lines.append(f"  {qtype:20s} n={m:3d}  accuracy={acc:.1%}  evidence_recall={rec:.1%}")

    if n and "cost_usd" in results[0]:
        total_wall_s = sum(r["wall_time_s"] for r in results)
        total_calls = sum(r["claude_calls"] for r in results)
        total_input = sum(r["input_tokens"] for r in results)
        total_output = sum(r["output_tokens"] for r in results)
        total_cache_read = sum(r["cache_read_tokens"] for r in results)
        total_cache_creation = sum(r["cache_creation_tokens"] for r in results)
        total_cost = sum(r["cost_usd"] for r in results)

        lines.append("\nEfficiency:")
        lines.append(f"  wall time: {total_wall_s:.1f}s total, {total_wall_s / n:.1f}s/question")
        lines.append(f"  claude CLI calls: {total_calls} total, {total_calls / n:.1f}/question")
        lines.append(
            f"  tokens: {total_input:,} in, {total_output:,} out, "
            f"{total_cache_read:,} cache-read, {total_cache_creation:,} cache-creation"
        )
        lines.append(f"  cost: ${total_cost:.4f} total, ${total_cost / n:.4f}/question")

    return "\n".join(lines)
