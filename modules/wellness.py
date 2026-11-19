"""Non-prescriptive educational wellness ideas, plans, and activity tracking."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


WELLNESS_GOALS = {
    "sleep": "Sleep routines",
    "movement": "Comfortable movement",
    "social": "Social connection",
    "mental": "Enjoyable mental activities",
    "wellbeing": "General wellbeing",
    "routine": "Healthy daily routines",
}

LIFESTYLE_OPTIONS = {
    "sleep_quality": (
        "Prefer not to say",
        "Usually restful",
        "Mixed",
        "Often difficult",
    ),
    "physical_activity": (
        "Prefer not to say",
        "Regular activity",
        "Some activity",
        "Limited",
    ),
    "social_engagement": (
        "Prefer not to say",
        "Frequently connected",
        "Some connection",
        "Would like more connection",
    ),
    "hydration": (
        "Prefer not to say",
        "Usually",
        "Sometimes",
        "Often forget",
    ),
    "mental_engagement": (
        "Prefer not to say",
        "Frequent",
        "Some",
        "Limited",
    ),
}

DAILY_ACTIVITIES = (
    {
        "id": "sleep_routine",
        "goal": "sleep",
        "title": "Choose a consistent sleep routine",
        "description": "Pick a wind-down or wake-up cue that suits your routine.",
    },
    {
        "id": "comfortable_movement",
        "goal": "movement",
        "title": "Add comfortable movement",
        "description": (
            "If safe and comfortable, try a short movement break at a pace that "
            "suits your abilities."
        ),
    },
    {
        "id": "social_connection",
        "goal": "social",
        "title": "Connect with someone",
        "description": (
            "Consider a conversation or message with someone you enjoy, if you wish."
        ),
    },
    {
        "id": "enjoyable_mental_activity",
        "goal": "mental",
        "title": "Choose an enjoyable mental activity",
        "description": (
            "Spend a few minutes on reading, music, a puzzle, a craft, or learning "
            "something you enjoy."
        ),
    },
    {
        "id": "restful_pause",
        "goal": "wellbeing",
        "title": "Take a restorative pause",
        "description": "Make room for a restful moment or a few comfortable slow breaths.",
    },
    {
        "id": "balanced_meal",
        "goal": "routine",
        "title": "Make time for a balanced meal or snack",
        "description": (
            "Choose food that fits your preferences and any guidance you already "
            "follow."
        ),
    },
    {
        "id": "plan_something_enjoyable",
        "goal": "routine",
        "title": "Plan one enjoyable activity",
        "description": "Choose one small activity that feels meaningful to you today.",
    },
)

DAILY_TIPS = (
    "A regular sleep and wake-up routine may make it easier to build a restful evening.",
    "Choose movement that feels safe and manageable for your abilities and circumstances.",
    "A small moment of social connection can be as simple as a message or conversation.",
    "Enjoyable activities such as reading, music, games, or crafts can add variety to a day.",
    "A balanced meal can be part of a routine that works for your preferences and needs.",
    "Short pauses and manageable routines can help make a busy day feel more approachable.",
    "There is no single routine that works for everyone; choose what feels supportive to you.",
)

_HABIT_IDEAS = {
    "sleep_quality": {
        "Often difficult": (
            "If useful, consider a consistent wind-down routine; bring persistent "
            "sleep concerns to a clinician."
        ),
        "Mixed": (
            "You could notice which everyday routines support restful sleep and "
            "share ongoing concerns with a clinician."
        ),
        "Usually restful": (
            "You report usually restful sleep; maintaining routines that work for "
            "you may be worthwhile."
        ),
    },
    "physical_activity": {
        "Limited": (
            "If safe and comfortable for you, consider small amounts of enjoyable "
            "movement; ask a clinician if you are unsure what is suitable."
        ),
        "Some activity": (
            "You could continue activities that feel manageable and enjoyable, "
            "at a pace that suits you."
        ),
        "Regular activity": (
            "You report regular movement; continue to choose activities that feel "
            "safe and sustainable for you."
        ),
    },
    "social_engagement": {
        "Would like more connection": (
            "If it feels right, consider a small, low-pressure way to connect, "
            "such as a message or shared activity."
        ),
        "Some connection": (
            "You could continue making time for social contact that feels comfortable "
            "and meaningful to you."
        ),
        "Frequently connected": (
            "You report frequent social contact; continue choosing connections "
            "that feel supportive to you."
        ),
    },
    "hydration": {
        "Often forget": (
            "A regular water break may be a helpful reminder, unless a clinician "
            "has advised you to limit fluids."
        ),
        "Sometimes": (
            "You could keep water accessible and use routine breaks as a reminder, "
            "unless you have been advised to limit fluids."
        ),
        "Usually": (
            "You report usually remembering hydration; continue to follow any "
            "individual fluid guidance from your clinician."
        ),
    },
    "mental_engagement": {
        "Limited": (
            "You might choose an enjoyable conversation, game, reading, music, or "
            "learning activity that fits your interests."
        ),
        "Some": (
            "You could continue enjoyable activities that involve conversation, "
            "learning, creativity, or problem-solving."
        ),
        "Frequent": (
            "You report frequent mental engagement; continue choosing activities "
            "you find meaningful and enjoyable."
        ),
    },
}


def suggested_activities(goals=()):
    """Return general activities, narrowed only by goals explicitly selected."""
    selected_goals = set(goals or ())
    unknown_goals = selected_goals - WELLNESS_GOALS.keys()
    if unknown_goals:
        raise ValueError(f"Unknown wellness goals: {', '.join(sorted(unknown_goals))}.")
    if not selected_goals:
        return [dict(activity) for activity in DAILY_ACTIVITIES]
    return [
        dict(activity)
        for activity in DAILY_ACTIVITIES
        if activity["goal"] in selected_goals
    ]


def daily_tip(for_date=None):
    """Return a stable, general educational tip for a calendar day."""
    current_date = for_date or date.today()
    return DAILY_TIPS[current_date.toordinal() % len(DAILY_TIPS)]


def normalize_wellness_state(payload=None):
    """Validate persisted plan data and return a safe, copy-on-write structure."""
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError("A saved wellness plan must be an object.")

    goals = payload.get("goals", [])
    if not isinstance(goals, list) or any(
        not isinstance(goal, str) or goal not in WELLNESS_GOALS
        for goal in goals
    ):
        raise ValueError("Saved wellness goals are invalid.")

    lifestyle = payload.get("lifestyle", {})
    if not isinstance(lifestyle, dict):
        raise ValueError("Saved wellness preferences must be an object.")
    normalized_lifestyle = {}
    for field, options in LIFESTYLE_OPTIONS.items():
        value = lifestyle.get(field, "Prefer not to say")
        if value not in options:
            raise ValueError(f"Saved wellness preference '{field}' is invalid.")
        normalized_lifestyle[field] = value

    completed_by_date = payload.get("completed_by_date", {})
    if not isinstance(completed_by_date, dict):
        raise ValueError("Saved activity history must be an object.")
    valid_activity_ids = {activity["id"] for activity in DAILY_ACTIVITIES}
    normalized_history = {}
    for day, activity_ids in completed_by_date.items():
        if not isinstance(day, str) or not isinstance(activity_ids, list):
            raise ValueError("Saved daily activity entries are invalid.")
        try:
            date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError("A saved activity date is invalid.") from exc
        if any(
            not isinstance(activity_id, str)
            or activity_id not in valid_activity_ids
            for activity_id in activity_ids
        ):
            raise ValueError("A saved activity identifier is invalid.")
        normalized_history[day] = list(dict.fromkeys(activity_ids))

    saved_at = payload.get("saved_at")
    if saved_at is not None and not isinstance(saved_at, str):
        raise ValueError("The wellness plan save date is invalid.")
    return {
        "version": 1,
        "goals": list(dict.fromkeys(goals)),
        "lifestyle": normalized_lifestyle,
        "completed_by_date": normalized_history,
        "saved_at": saved_at,
    }


def wellness_progress_rows(completed_by_date, today=None, days=7):
    """Return recent logged completions, leaving untracked days missing."""
    if isinstance(days, bool) or not isinstance(days, int) or days < 1:
        raise ValueError("The progress window must be a positive number of days.")
    end_date = today or date.today()
    rows = []
    for offset in range(days - 1, -1, -1):
        day = (end_date - timedelta(days=offset)).isoformat()
        activities = completed_by_date.get(day)
        rows.append({
            "Date": day,
            "Activities marked complete": len(activities) if activities is not None else None,
        })
    return rows


def build_wellness_plan(completed_domains, lifestyle, goals=()):
    """Build optional educational ideas from self-report and user-selected goals."""
    lifestyle = lifestyle or {}
    if not isinstance(lifestyle, dict):
        raise ValueError("Lifestyle responses must be an object.")
    selected_goals = list(dict.fromkeys(goals or ()))
    activities = suggested_activities(selected_goals)
    habit_ideas = []
    for field, options in _HABIT_IDEAS.items():
        response = lifestyle.get(field)
        if response in options:
            habit_ideas.append(options[response])

    domains = sorted(set(completed_domains))
    checklist = [
        "Describe any health concerns, when they began, and how they affect daily life.",
        "Bring an up-to-date list of medicines and supplements.",
        "Ask which assessments or follow-up, if any, are appropriate for your situation.",
    ]
    if domains:
        checklist.insert(
            0,
            "Review the recorded activity coverage: " + ", ".join(domains) + ".",
        )
    else:
        checklist.insert(
            0,
            "No completed assessment domains are recorded in saved sessions to review.",
        )
    checklist.append(
        "Discuss any device, language, fatigue, or accessibility factors that may "
        "have affected measurements."
    )
    return {
        "goals": selected_goals,
        "habit_ideas": habit_ideas,
        "suggested_activities": activities,
        "clinician_checklist": checklist,
    }


def new_saved_wellness_state(goals, lifestyle, completed_by_date):
    """Create a validated persisted snapshot after an explicit user save."""
    state = normalize_wellness_state({
        "goals": list(goals),
        "lifestyle": dict(lifestyle),
        "completed_by_date": completed_by_date,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    })
    return state
