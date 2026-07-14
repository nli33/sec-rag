"""M3: aggregate per-question eval results into a metrics report."""
from collections import defaultdict


def build_report(results: list[dict]) -> str:
    """`results` items: {question_type, correct, evidence_recall}. Returns a printable report."""
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

    return "\n".join(lines)
