"""Signal confidence scoring: quality/usability only, never disease likelihood."""
def confidence_report(reasons, checks_passed, checks_total):
    reasons = list(dict.fromkeys(reasons))
    if checks_total <= 0:
        score = 0
    else:
        score = round(100 * max(0, min(checks_passed, checks_total)) / checks_total)
    if reasons:
        status = "repeat_recommended"
        label = "Review signal quality"
    elif score >= 80:
        status = "usable"
        label = "Basic quality checks passed"
    else:
        status = "limited"
        label = "Limited quality evidence"
    return {
        "score": score,
        "status": status,
        "label": label,
        "reasons": reasons,
        "meaning": "This score describes technical signal quality only; it is not a health or disease-risk score.",
    }
