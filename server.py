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
    # Added explicit headers for proxy compatibility as requested
    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


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
            yield "data: No weak areas detected. Keep it up!\n\n"
            yield "data: [DONE]\n\n"
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
        return StreamingResponse(empty(), media_type="text/event-stream", headers=headers)

    system = WEAK_AREA_SYSTEM.format(weak_topics=", ".join(req.weak_topics))
    history = [{"role": "user", "content": "Start my targeted review session."}]
    return sse(stream_tokens(client, system, history))


# ── Run ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
