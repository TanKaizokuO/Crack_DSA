# DSA Coach — Python + Web UI Rebuild Spec
> Paste this file into Claude Code or Aider. It is a complete, self-contained implementation spec.

---

## Goal
Build a **web UI-based** interactive DSA coach in Python that follows **Striver's A2Z DSA Sheet**.  
It uses **DeepSeek via NVIDIA NIM** as the LLM backend and replaces the original CLI with a browser-based interface served by a local FastAPI server.

Features:
- Streaming responses (server-sent events to browser)
- Spaced-repetition scheduling
- Hint system (3-tier)
- Rich dashboard with progress charts
- Conversation memory within a session
- Persistent weak-area review queue

---

## Tech Stack

```
python >= 3.11
fastapi
uvicorn[standard]
langchain-nvidia-ai-endpoints
langchain-core
python-dotenv
```

`requirements.txt`:
```
fastapi
uvicorn[standard]
langchain-nvidia-ai-endpoints
langchain-core
python-dotenv
```

> No `inquirer` or `rich` — all UI is handled in the browser.

---

## LLM Client — copy exactly

```python
# llm.py
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Generator

def get_client() -> ChatNVIDIA:
    return ChatNVIDIA(
        model=os.environ.get("NVIDIA_MODEL", "deepseek-ai/deepseek-r1-zero"),
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": "high"
            }
        },
    )

def stream_tokens(client: ChatNVIDIA, system: str, history: list[dict]) -> Generator[str, None, None]:
    """
    Yields string tokens one at a time.
    history: list of {"role": "user"|"assistant", "content": str}
    """
    messages = [SystemMessage(content=system)]
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    for chunk in client.stream(messages):
        if chunk.content:
            yield chunk.content
```

---

## Project Structure

```
dsa-coach/
├── server.py           # FastAPI entry point
├── llm.py              # LLM client + streaming helper
├── modes/
│   ├── __init__.py
│   ├── learn.py        # teaching mode logic
│   ├── practice.py     # evaluation + hint logic
│   └── review.py       # weak-area review logic
├── state/
│   ├── __init__.py
│   ├── progress.py     # load/save progress.json
│   └── scheduler.py    # spaced-repetition logic (SM-2 lite)
├── static/
│   └── app.html        # single-file frontend (HTML + CSS + JS)
├── curriculum.json     # copy unchanged from original project
├── progress.json       # auto-created on first run
├── sessions/           # auto-created directory
└── .env                # NVIDIA_API_KEY=...
```

---

## System Prompts

Define these as module-level constants in `modes/prompts_text.py`:

```python
TEACH_SYSTEM = """
You are an expert DSA coach teaching a software engineer preparing for FAANG interviews.

Rules:
- NEVER write code. Explain using plain English, pseudocode, and ASCII diagrams only.
- Start with intuition and real-world analogy before going technical.
- Cover: concept definition, time complexity, space complexity, common patterns, pitfalls.
- End every teaching session with 2-3 self-check questions the user should be able to answer.
- Be concise but thorough. Use bullet points and ASCII art liberally.
"""

TEACH_QA_SYSTEM = """
You are a DSA tutor answering follow-up questions during a teaching session.
The user is learning {topic}. Answer their question clearly and briefly.
Never write executable code. Use pseudocode and diagrams only.
If the question is off-topic, gently redirect.
"""

EVAL_SYSTEM = """
You are a senior FAANG interviewer evaluating a candidate's problem-solving approach.

Problem: {problem_title} (difficulty: {difficulty})

Evaluation rules:
- The user will describe their approach in plain English.
- Respond with a structured evaluation block:
  VERDICT: CORRECT | PARTIALLY CORRECT | WRONG
  SCORE: 0-100
  STRENGTHS: (bullet list)
  GAPS: (bullet list)
  FOLLOWUP: (one targeted follow-up question)
- Be strict but fair. Partial credit for right intuition with wrong complexity.
- Never give away the full solution. Guide via questions.
"""

HINT_SYSTEM = """
You are a DSA coach giving a hint for: {problem_title}

Hint level {hint_level}/3:
  Level 1 — nudge toward the right data structure or pattern only.
  Level 2 — describe the high-level approach without details.
  Level 3 — walk through the algorithm step-by-step in pseudocode.

Give ONLY the hint for the requested level. Do not reveal code.
"""

FOLLOWUP_SYSTEM = """
You are a FAANG interviewer doing a deep-dive on: {problem_title}

The candidate gave this approach: {user_approach}
Your previous evaluation: {previous_eval}

Ask one specific follow-up that probes:
- Edge cases they may have missed
- How they'd handle scale (10x data)
- Alternative approaches and their trade-offs

Keep it to 2-3 sentences max.
"""

WEAK_AREA_SYSTEM = """
You are a DSA coach reviewing a student's weak areas.
The student has scored below 60% on: {weak_topics}

Craft a targeted 5-minute review session that:
1. Re-explains the core concept in a fresh way (different analogy than before)
2. Gives one warm-up problem statement (no solution)
3. Asks them to identify the pattern

Do not write code.
"""
```

---

## `state/progress.py`

```python
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
```

---

## `state/scheduler.py` — SM-2 lite

```python
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
```

---

## `server.py` — FastAPI Entry Point

```python
#!/usr/bin/env python3
"""
DSA Coach — Web UI server
Run: uvicorn server:app --reload --port 8000
Then open: http://localhost:8000
"""

import json
import re
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from llm import get_client, stream_tokens
from state import progress as state_mod
from modes.prompts_text import (
    TEACH_SYSTEM, TEACH_QA_SYSTEM, EVAL_SYSTEM,
    HINT_SYSTEM, FOLLOWUP_SYSTEM, WEAK_AREA_SYSTEM
)

app = FastAPI(title="DSA Coach")
SESSIONS_DIR = Path("sessions")
SESSIONS_DIR.mkdir(exist_ok=True)

client = get_client()


# ── Pydantic models ──────────────────────────────────────────────────

class LearnRequest(BaseModel):
    subtopic_title: str
    concepts: list[str]
    problem_titles: list[str]
    history: list[dict] = []

class QARequest(BaseModel):
    topic: str
    question: str
    history: list[dict]

class EvalRequest(BaseModel):
    problem_title: str
    difficulty: str
    approach: str
    history: list[dict] = []

class HintRequest(BaseModel):
    problem_title: str
    hint_level: int  # 1-3

class ReviewRequest(BaseModel):
    weak_topics: list[str]

class MarkLearnedRequest(BaseModel):
    subtopic_id: str

class RecordAttemptRequest(BaseModel):
    problem_id: str
    subtopic_id: str
    score: int
    session_log: list[dict] = []


# ── Static files + SPA ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html = Path("static/app.html").read_text()
    return HTMLResponse(html)


# ── Curriculum ───────────────────────────────────────────────────────

@app.get("/api/curriculum")
async def get_curriculum():
    p = Path("curriculum.json")
    if not p.exists():
        return JSONResponse({"error": "curriculum.json not found"}, status_code=404)
    return json.loads(p.read_text())


# ── Progress ─────────────────────────────────────────────────────────

@app.get("/api/progress")
async def get_progress():
    return state_mod.load()

@app.post("/api/progress/mark-learned")
async def mark_learned(req: MarkLearnedRequest):
    prog = state_mod.load()
    prog["subtopics_learned"][req.subtopic_id] = {
        "learned_at": datetime.now().isoformat(),
        "review_count": 0
    }
    state_mod.save(prog)
    return {"ok": True}

@app.post("/api/progress/record-attempt")
async def record_attempt(req: RecordAttemptRequest):
    prog = state_mod.load()
    state_mod.record_problem_attempt(prog, req.problem_id, req.score)
    state_mod.update_score(prog, req.subtopic_id, req.score)
    state_mod.save(prog)

    # Save session log
    if req.session_log:
        fname = SESSIONS_DIR / f"{req.subtopic_id}_{datetime.now():%Y%m%d_%H%M%S}.json"
        fname.write_text(json.dumps(req.session_log, indent=2))

    return {"ok": True}


# ── Streaming LLM endpoints ──────────────────────────────────────────

def sse(gen):
    """Wrap a token generator as a Server-Sent Events StreamingResponse."""
    def event_stream():
        for token in gen:
            escaped = token.replace("\n", "\\n")
            yield f"data: {escaped}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/learn/start")
async def learn_start(req: LearnRequest):
    prompt = (
        f"Teach me: {req.subtopic_title}\n"
        f"Key concepts to cover: {', '.join(req.concepts)}\n"
        f"Problems this applies to: {', '.join(req.problem_titles)}"
    )
    history = req.history + [{"role": "user", "content": prompt}]
    return sse(stream_tokens(client, TEACH_SYSTEM, history))


@app.post("/api/learn/qa")
async def learn_qa(req: QARequest):
    system = TEACH_QA_SYSTEM.format(topic=req.topic)
    history = req.history + [{"role": "user", "content": req.question}]
    return sse(stream_tokens(client, system, history))


@app.post("/api/practice/evaluate")
async def evaluate(req: EvalRequest):
    system = EVAL_SYSTEM.format(
        problem_title=req.problem_title,
        difficulty=req.difficulty
    )
    history = req.history + [{"role": "user", "content": req.approach}]
    return sse(stream_tokens(client, system, history))


@app.post("/api/practice/hint")
async def hint(req: HintRequest):
    system = HINT_SYSTEM.format(
        problem_title=req.problem_title,
        hint_level=req.hint_level
    )
    history = [{"role": "user", "content": f"Give me hint level {req.hint_level}."}]
    return sse(stream_tokens(client, system, history))


@app.post("/api/review/start")
async def review_start(req: ReviewRequest):
    if not req.weak_topics:
        async def empty():
            yield "data: No weak areas detected. Keep it up!\\n\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    system = WEAK_AREA_SYSTEM.format(weak_topics=", ".join(req.weak_topics))
    history = [{"role": "user", "content": "Start my targeted review session."}]
    return sse(stream_tokens(client, system, history))


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
```

---

## `static/app.html` — Single-File Frontend

This is the complete browser UI. It talks to the FastAPI backend over `fetch` and SSE.

### Layout

The app has a **persistent left sidebar** and a **main content area** that switches between four views:

| View | Description |
|------|-------------|
| **Dashboard** | Stats grid, per-step progress cards with score bars, weak-area panel |
| **Learn** | Curriculum tree sidebar + chat window with streaming AI responses |
| **Practice** | Problem list sidebar + approach textarea + streaming evaluation panel |
| **Review** | Weak-area chips + streaming review session output |

### Sidebar

```
┌─────────────────────┐
│  DSA Coach          │  ← logo + model badge
│  Striver's A2Z      │
├─────────────────────┤
│  ▣  Dashboard       │
│  📚  Learn          │
│  🎯  Practice       │
│  🔄  Review         │
├─────────────────────┤
│  🔥 streak: N days  │  ← bottom of sidebar
└─────────────────────┘
```

### Dashboard View

```
┌──────────┬──────────┬──────────┬──────────┐
│ Learned  │ Problems │ Avg Score│ Weak     │
│    12    │    24    │   73%    │    3     │
└──────────┴──────────┴──────────┴──────────┘

[ Step 1: Basics       ████████░░  80%  4/6  ✅ Strong  ]
[ Step 2: Sorting      █████░░░░░  52%  2/4  ⚠ Weak    ]
[ Step 3: Arrays       ███████░░░  70%  5/8  🔄 Review  ]
...

⚠ Weak Areas: Step 2, Step 5
```

Clicking a topic card navigates to Learn with that step pre-expanded.

### Learn View

```
┌──────────────────┬────────────────────────────────┐
│ 1. Basics      ▶ │  AI Coach                       │
│   ● Complexity   │  ┌─────────────────────────────┐│
│   ● Patterns     │  │ Here's how Big O works...   ││
│ 2. Sorting     ▶ │  └─────────────────────────────┘│
│   ○ Bubble       │                                  │
│   ○ Merge        │  Me                              │
│ 3. Arrays      ▶ │  ┌─────────────────────────────┐│
│   ○ Easy         │  │ What about O(n log n)?      ││
│   ○ Medium       │  └─────────────────────────────┘│
│                  │                                  │
│                  │ [💡 Analogy][⚠ Pitfalls][⏱ TC] │
│                  │ ┌──────────────────────┐ [Send] │
│                  │ │ Ask a follow-up...   │        │
│                  │ └──────────────────────┘        │
└──────────────────┴────────────────────────────────┘
```

- Clicking a subtopic calls `POST /api/learn/start` and streams the response into the chat.
- Quick-action buttons prepopulate the input and auto-send.
- The first AI response marks the subtopic as learned via `POST /api/progress/mark-learned`.
- The curriculum tree dots update: grey = not started, green = learned.

### Practice View

```
┌──────────────────┬────────────────────────────────┐
│ Problems         │ Two Sum                [Easy]   │
│ ─────────────── │ LC #1 · 0 attempts              │
│ Two Sum  [Easy]  │                                  │
│ Max Sub  [Med]   │ Your Approach                    │
│ 3Sum     [Med]   │ ┌──────────────────────────────┐│
│ Rain Wat [Hard]  │ │ I would use a hash map...    ││
│ ...              │ └──────────────────────────────┘│
│                  │ [Evaluate] [Hint 1][Hint 2][Hint3│
│                  │                                  │
│                  │ ── Evaluation ──────────────────│
│                  │ VERDICT: CORRECT    Score: [85] │
│                  │ ┌──────────────────────────────┐│
│                  │ │ STRENGTHS: ...               ││
│                  │ │ GAPS: ...                    ││
│                  │ │ FOLLOWUP: ...               ││
│                  │ └──────────────────────────────┘│
│                  │ [Try Again]  [Next Problem →]   │
└──────────────────┴────────────────────────────────┘
```

- Clicking a problem populates the header with its title and difficulty.
- "Evaluate" calls `POST /api/practice/evaluate` and streams the result.
- The app parses `SCORE: NN` and `VERDICT: ...` from the streamed text to update the score ring and badge in real time once the stream ends.
- Hint buttons call `POST /api/practice/hint` with the appropriate level.
- On completion, `POST /api/progress/record-attempt` is called to persist the score.

### Review View

```
Weak Area Review

Targeting: [Step 2] [Step 5]

[Start Review]

─────────────────────────────────────────
Here's a fresh way to think about sorting...

Warm-up: Given an array of 0s, 1s, and 2s...

Can you identify the pattern?
─────────────────────────────────────────
```

Calls `POST /api/review/start` with the list of weak topic IDs and streams the session.

### SSE Streaming Pattern (JavaScript)

All streaming endpoints return `text/event-stream`. Use this pattern for every stream:

```javascript
async function streamSSE(url, body, onToken, onDone) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();                   // keep incomplete line
    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;
      const data = line.slice(6);
      if (data === "[DONE]") { onDone?.(); return; }
      onToken(data.replace(/\\n/g, "\n")); // unescape newlines
    }
  }
}
```

### Design Spec

Implement `static/app.html` as a **single self-contained file** (HTML + CSS + JS, no external dependencies except Google Fonts).

Visual language:
- **Dark theme** — background `#0d0f14`, surfaces `#141720` / `#1c2030`
- **Accent colours** — teal `#5eead4` (primary), indigo `#818cf8` (user), orange `#fb923c` (streak/hints)
- **Typography** — [Sora](https://fonts.google.com/specimen/Sora) for UI, [Space Mono](https://fonts.google.com/specimen/Space+Mono) for code/numbers/badges
- **Borders** — `1px solid rgba(255,255,255,0.07)` (default), `rgba(255,255,255,0.13)` (hover/focus)
- **Progress bars** — 4px tall, green ≥80%, yellow 60–79%, red <60%
- **Score ring** — SVG circle with `stroke-dashoffset` animated from 138.2 (0%) to 0 (100%)
- **Difficulty badges** — pill shape; Easy=green, Medium=yellow, Hard=red, all with 12% opacity fill

---

## `.env`

```
NVIDIA_API_KEY=nvapi-...
NVIDIA_MODEL=deepseek-ai/deepseek-r1-zero
```

---

## Bootstrap Commands

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Add your key
echo "NVIDIA_API_KEY=nvapi-..." > .env

# 3. Copy curriculum.json from the original project (no changes needed)

# 4. Run
uvicorn server:app --reload --port 8000

# 5. Open browser
open http://localhost:8000
```

---

## What Changed vs CLI Version

| Feature | CLI version | This version (Web UI) |
|---|---|---|
| Interface | Terminal (`inquirer`, `rich`) | Browser (`static/app.html`) |
| LLM transport | LangChain stream → stdout | LangChain stream → SSE → browser |
| Navigation | Arrow-key menus | Sidebar nav + clickable lists |
| Progress display | `rich` table on request | Always-visible dashboard |
| Streaming | Prints tokens to terminal | Tokens append into chat bubbles |
| State persistence | `progress.json` (CLI reads) | `progress.json` (API reads/writes) |
| Hints | Type `hint` at prompt | Click Hint 1 / 2 / 3 buttons |
| Score ring | N/A | Animated SVG ring |
| Spaced repetition | SM-2 scheduler | SM-2 scheduler (unchanged) |
| Session logs | Written to `sessions/` JSON | Written to `sessions/` JSON (same) |
| Entry point | `python coach.py` | `uvicorn server:app --port 8000` |

---

## Notes for Claude Code / Aider

- Implement all files exactly as specified.
- Create `modes/__init__.py`, `state/__init__.py` as empty files.
- `sessions/` is created automatically by `server.py` on startup.
- `progress.json` is created automatically by `state/progress.py` on first write.
- Do not modify `curriculum.json`.
- `static/app.html` must be a **single file** — no separate CSS or JS files.
- All four views (Dashboard, Learn, Practice, Review) must be implemented with working API calls.
- If `ChatNVIDIA` raises a model-not-found error at startup, call `ChatNVIDIA.get_available_models()`, print the list, and exit with a helpful message.
- SSE endpoints must set `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers for proxy compatibility.