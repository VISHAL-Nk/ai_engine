"""
Trend Detector Module
Extracts sliding-window trend detection logic from the router.
"""
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

def compute_windows(reviews: list[dict], window_size: int) -> list[list[dict]]:
    total = len(reviews)
    windows: list[list[dict]] = []
    for i in range(0, total, window_size):
        chunk = reviews[i : i + window_size]
        if len(chunk) >= window_size // 2:
            windows.append(chunk)
    return windows

def compute_feature_timeline(windows: list[list[dict]]) -> dict[str, list[dict]]:
    feature_timeline: dict[str, list[dict]] = defaultdict(list)
    for win_idx, window in enumerate(windows):
        feature_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "neutral": 0, "ambiguous": 0, "total": 0}
        )
        for review in window:
            for fs in review.get("featureSentiments", []):
                attr = fs.get("attribute", "unknown")
                sent = fs.get("sentiment", "neutral")
                feature_counts[attr][sent] += 1
                feature_counts[attr]["total"] += 1

        for feature, counts in feature_counts.items():
            t = counts["total"]
            if t == 0:
                continue
            feature_timeline[feature].append({
                "window": win_idx + 1, # 1-indexed for display
                "positive": round(counts["positive"] / t, 3),
                "negative": round(counts["negative"] / t, 3),
                "neutral": round(counts["neutral"] / t, 3),
                "total_mentions": t,
            })
    return feature_timeline

def analyze_trends(windows: list[list[dict]], feature_timeline: dict[str, list[dict]], window_size: int) -> list[dict]:
    trend_alerts = []
    if len(windows) < 2:
        return trend_alerts

    current_window = windows[-1]
    previous_window = windows[-2]

    for feature, timeline in feature_timeline.items():
        if len(timeline) < 2:
            continue

        current = timeline[-1]
        previous = timeline[-2]

        current_neg = current["negative"]
        previous_neg = previous["negative"]
        change = current_neg - previous_neg

        if change > 0.15:
            # Rising complaint
            unique_reviewers = set()
            examples = []
            for review in current_window:
                for fs in review.get("featureSentiments", []):
                    if fs.get("attribute") == feature and fs.get("sentiment") == "negative":
                        unique_reviewers.add(str(review.get("customerId", "")))
                        if len(examples) < 3:
                            examples.append(review.get("text", "")[:120])
            
            n_reviewers = len(unique_reviewers)
            severity = "isolated"
            if n_reviewers > 4:
                severity = "systemic"
            elif n_reviewers > 2:
                severity = "emerging"

            trend_alerts.append({
                "feature": feature,
                "direction": "rising_complaint",
                "severity": severity,
                "current_ratio": round(current_neg, 3),
                "previous_ratio": round(previous_neg, 3),
                "change_pct": round(change * 100, 1),
                "window_description": f"last {window_size} reviews vs previous {window_size}",
                "unique_reviewers": n_reviewers,
                "example_reviews": examples,
            })

        current_pos = current["positive"]
        previous_pos = previous["positive"]
        if previous_pos > 0.80 and current_pos < 0.50:
            # Sudden drop
            trend_alerts.append({
                "feature": feature,
                "direction": "sudden_drop",
                "severity": "systemic",
                "current_ratio": round(current_pos, 3),
                "previous_ratio": round(previous_pos, 3),
                "change_pct": round((previous_pos - current_pos) * 100, 1),
                "window_description": f"last {window_size} reviews vs previous {window_size}",
                "unique_reviewers": 0,
                "example_reviews": [],
            })

    return trend_alerts
