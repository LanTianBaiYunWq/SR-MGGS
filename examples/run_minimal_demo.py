"""CPU-only smoke test for SR-MGGS score-level authentication.

This script validates the released composite score, max-over-K aggregation,
independent FAR threshold calibration, and accept/reject decision. It does not
replace end-to-end feature extraction from vector GIS data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List


def l2_normalize(vector: Iterable[float]) -> List[float]:
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return [value / norm for value in values]


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_normalized = l2_normalize(left)
    right_normalized = l2_normalize(right)
    if len(left_normalized) != len(right_normalized):
        raise ValueError("Signature dimensions do not match.")
    return sum(a * b for a, b in zip(left_normalized, right_normalized))


def composite_score(
    query: Dict[str, Any],
    template: Dict[str, Any],
    point_alpha: float,
    generic_beta: float,
) -> float:
    score = cosine_similarity(query["lp"], template["lp"])
    query_point = query.get("point")
    template_point = template.get("point")
    if query_point is not None and template_point is not None:
        score += point_alpha * cosine_similarity(query_point, template_point)
    score += generic_beta * cosine_similarity(query["generic"], template["generic"])
    return float(score)


def higher_quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Calibration scores are empty.")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1].")
    index = int(math.ceil(quantile * (len(ordered) - 1)))
    return ordered[index]


def run_demo(payload: Dict[str, Any]) -> Dict[str, Any]:
    target_far = float(payload["target_far"])
    threshold = higher_quantile(payload["calibration_impostor_scores"], 1.0 - target_far)
    point_alpha = float(payload["point_alpha"])
    generic_beta = float(payload["generic_beta"])
    query = payload["query"]

    candidate_scores: Dict[str, float] = {}
    template_scores: Dict[str, List[float]] = {}
    for source_id, templates in payload["registered_sources"].items():
        scores = [
            composite_score(query, template, point_alpha=point_alpha, generic_beta=generic_beta)
            for template in templates
        ]
        if not scores:
            raise ValueError(f"Source {source_id!r} has no templates.")
        template_scores[source_id] = scores
        candidate_scores[source_id] = max(scores)

    predicted_source = max(candidate_scores, key=candidate_scores.get)
    authentication_score = candidate_scores[predicted_source]
    decision = "Accept" if authentication_score >= threshold else "Reject"
    return {
        "target_far": target_far,
        "calibrated_threshold": threshold,
        "point_alpha": point_alpha,
        "generic_beta": generic_beta,
        "templates_per_source": {
            source_id: len(scores) for source_id, scores in template_scores.items()
        },
        "template_scores": template_scores,
        "candidate_scores": candidate_scores,
        "predicted_source": predicted_source,
        "authentication_score": authentication_score,
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = run_demo(payload)
    print(f"Calibrated threshold: {result['calibrated_threshold']:.6f}")
    for source_id, score in sorted(result["candidate_scores"].items()):
        print(f"Candidate {source_id}: {score:.6f}")
    print(f"Predicted source: {result['predicted_source']}")
    print(f"Authentication score: {result['authentication_score']:.6f}")
    print(f"Decision: {result['decision']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

