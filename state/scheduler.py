from datetime import datetime, timedelta

def next_review_days(score: int, interval: int = 1, ease: float = 2.5) -> int:
    q = score / 20
    if q < 3:
        return 1
    new_ease = max(1.3, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    return max(1, round(interval * new_ease))

def is_due(last_review_iso: str | None, interval_days: int) -> bool:
    if last_review_iso is None:
        return True
    last = datetime.fromisoformat(last_review_iso)
    return datetime.now() >= last + timedelta(days=interval_days)
