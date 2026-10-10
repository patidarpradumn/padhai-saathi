"""Rule-based recommender: turns model predictions into concrete actions.

This is the 'decide + act' step of the agent loop:
  perceive (student data) -> predict (ML models) -> decide (these rules) -> act (study plan / teacher alert)
Every rule is transparent so a teacher can see WHY an action was suggested.
"""

HIGH, MED, LOW = "High", "Medium", "Low"


def recommend(s: dict, g3_pred: float, passed: int, risk: str):
    """s: dict with keys studytime, failures, absences, G1, G2, goout, Dalc, freetime, health."""
    acts = []

    def add(priority, action, why):
        acts.append({"priority": priority, "action": action, "why": why})

    if s["failures"] >= 1:
        add(HIGH if (not passed or g3_pred < 12) else MED, "Revise last year's prerequisite topics; teacher to share a short bridge-topic list.",
            f"{int(s['failures'])} past class failure(s) - gaps in basics carry forward.")
    if s["G2"] < 10:
        add(HIGH, "Pick the 3 weakest topics from the last test and practise 10 questions on each, every day.",
            f"Latest period grade is {s['G2']:.0f}/20, below the pass mark.")
    if s["G2"] < s["G1"] - 1:
        add(HIGH, "Review mistakes from the most recent unit test with the subject teacher this week.",
            f"Grade fell from {s['G1']:.0f} to {s['G2']:.0f} between periods.")
    if s["absences"] > 10:
        add(MED, "Assign a classmate 'notes buddy' and catch up on missed lessons every weekend.",
            f"{int(s['absences'])} absences - missed lessons build up.")
    if s["studytime"] <= 1:
        add(MED, "Fix a daily 30-minute study slot at the same time each day.",
            "Reported weekly study time is in the lowest band.")
    if s["goout"] >= 4 and s["studytime"] <= 2:
        add(MED, "Keep weekday evenings for study; plan outings only on weekends.",
            "High outings combined with low study time.")
    if s["health"] <= 2 or s["Dalc"] >= 3:
        add(MED, "Refer for a friendly check-in with a teacher/counsellor.",
            "Low health score or high weekday alcohol use - wellbeing may be affecting studies.")
    if passed and 10 <= g3_pred < 12:
        add(MED, "Borderline pass predicted: add one weekly revision test to stay above the line.",
            f"Predicted final grade {g3_pred:.1f}/20 is close to the pass mark.")
    if not acts:
        add(LOW, "On track. Keep the current routine and try one stretch problem set per week.",
            "No risk rules triggered.")

    order = {HIGH: 0, MED: 1, LOW: 2}
    acts.sort(key=lambda a: order[a["priority"]])
    return acts


def teacher_alert(name: str, g3_pred: float, passed: int, risk: str):
    if not passed or risk == "High risk":
        return f"ALERT: {name} - predicted G3 {g3_pred:.1f}/20, {risk}. Needs follow-up this week."
    return ""
