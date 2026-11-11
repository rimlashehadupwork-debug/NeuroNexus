from modules.confidence import confidence_report
def test_quality_score_is_not_risk_score():
    result = confidence_report([], 5, 5)
    assert result["score"] == 100
    assert "not a health or disease-risk score" in result["meaning"]
def test_reasons_trigger_review():
    result = confidence_report(["too short"], 4, 5)
    assert result["status"] == "repeat_recommended"
