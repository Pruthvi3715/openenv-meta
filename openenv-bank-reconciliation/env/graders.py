from typing import Dict, List, Any
from rapidfuzz import fuzz

from .models import State, MERCHANT_CATEGORIES


def grade_task1(resolved: Dict[str, dict], state: State) -> float:
    if not resolved:
        return 0.0

    correct = 0
    total = len(resolved)

    for tid, resolution in resolved.items():
        ground_truth = state.ground_truth_categories.get(tid, "Unknown")
        if resolution.get("category") == ground_truth:
            correct += 1

    return correct / total if total > 0 else 0.0


def grade_task2(resolved: Dict[str, dict], state: State) -> float:
    if not resolved:
        return 0.0

    scores = []
    total = len(resolved)

    for tid, resolution in resolved.items():
        ground_truth_merchant = state.ground_truth_merchants.get(tid, "")
        merchant_label = resolution.get("merchant_label", "")

        if ground_truth_merchant and merchant_label:
            score = (
                fuzz.ratio(merchant_label.lower(), ground_truth_merchant.lower())
                / 100.0
            )
            scores.append(score)

    return sum(scores) / len(scores) if scores else 0.0


def grade_task3(resolved: Dict[str, dict], state: State) -> float:
    if not resolved:
        return 0.0

    total = len(state.all_transactions)
    correct_categories = 0

    for t in state.all_transactions:
        tid = t.id
        if tid in resolved:
            ground_truth = state.ground_truth_categories.get(tid, "Unknown")
            if resolved[tid].get("category") == ground_truth:
                correct_categories += 1

    category_score = correct_categories / total if total > 0 else 0.0

    flagged_pairs = set()
    for tid, resolution in resolved.items():
        if resolution.get("flag_type") == "duplicate":
            for dup_pair in state.duplicates:
                if tid in dup_pair:
                    flagged_pairs.add(dup_pair)
                    break

    flagged_duplicates = len(flagged_pairs)
    duplicate_score = (
        flagged_duplicates / len(state.duplicates) if state.duplicates else 1.0
    )

    flagged_anomalies = 0
    for tid, resolution in resolved.items():
        if resolution.get("flag_type") == "anomaly" and tid in state.anomalies:
            flagged_anomalies += 1

    anomaly_score = flagged_anomalies / 2 if state.anomalies else 1.0

    final_score = round(
        0.4 * category_score + 0.3 * duplicate_score + 0.3 * anomaly_score, 4
    )

    return final_score


def grade_task(task_name: str, resolved: Dict[str, dict], state: State) -> Any:
    if task_name == "categorize":
        return grade_task1(resolved, state)
    elif task_name == "decode_upi":
        return grade_task2(resolved, state)
    elif task_name == "full_reconciliation":
        return grade_task3(resolved, state)
    else:
        raise ValueError(f"Unknown task: {task_name}")
