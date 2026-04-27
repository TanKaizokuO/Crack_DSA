import json
from pathlib import Path
from datetime import datetime

PROGRESS_FILE = Path("progress.json")

DEFAULT_PROGRESS = {
    "subtopics_learned": {},     # topic_id -> {"learned_at": iso, "review_count": int}
    "problems": {},              # problem_id -> {"attempts": int, "best_score": int, "last_attempt": iso}
    "scores": {},                # topic_id -> float (0-100)
    "review_queue": [],          # list of topic_ids scheduled for review
    "streak": 0,
    "last_session_date": None,
}

def load() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return DEFAULT_PROGRESS.copy()

def save(data: dict):
    data["last_session_date"] = datetime.now().isoformat()
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))

def get_weak_topics(progress: dict, threshold: int = 60) -> list[str]:
    return [t for t, s in progress["scores"].items() if s < threshold]

def update_score(progress: dict, topic_id: str, new_score: int):
    existing = progress["scores"].get(topic_id)
    if existing is None:
        progress["scores"][topic_id] = new_score
    else:
        progress["scores"][topic_id] = round(0.4 * existing + 0.6 * new_score, 1)

def record_problem_attempt(progress: dict, problem_id: str, score: int):
    entry = progress["problems"].setdefault(problem_id, {
        "attempts": 0, "best_score": 0, "last_attempt": None
    })
    entry["attempts"] += 1
    entry["best_score"] = max(entry["best_score"], score)
    entry["last_attempt"] = datetime.now().isoformat()
