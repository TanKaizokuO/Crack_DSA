# Crack DSA

## Overview
**Vision/Goal:** A full-stack AI coaching platform that simulates real FAANG-style interview preparation by enforcing a structured learning loop (Understand → Think → Explain → Validate → Improve) rather than just providing solutions.

**Current Status:** Active Development

## Tech Stack
**Language/Runtime:** Python 3.x, JavaScript (Vanilla)

**Frameworks/Libraries:** FastAPI (Backend), Vanilla HTML/CSS/JS (Frontend SPA)

**Key Dependencies:** LangChain, Nvidia AI Endpoints (Model: `moonshotai/kimi-k2-instruct-0905`), E2B Sandbox (for safe code execution).

## Directory Structure
```text
Crack_DSA/
├── curriculum.json      # Structured DSA curriculum data
├── llm.py               # LLM LangChain interface and token streaming logic
├── modes/               # Prompt templates/constraints for different coaching modes
├── progress.json        # User progress persistence
├── server.py            # FastAPI application, routing, and SSE streaming
├── sessions/            # Saved JSON logs of user session attempts
├── state/               # State management logic (e.g., progress.py)
└── static/              # Frontend assets (Vanilla JS SPA in app.html)
```

## Core Logic & Data Flow
1. **Interactive Coaching Loop:** The Vanilla JS frontend (SPA) sends user input (topic selection or problem approach) to the FastAPI backend (`server.py`). The backend determines the context (Teaching, Mock Interview, Hints) and builds a mode-specific prompt using templates from `modes/`.
2. **Streaming AI Responses:** The backend passes the constrained prompt to `llm.py`, which communicates with the Nvidia LangChain endpoint. The LLM generates a reasoning-focused response (without full code solutions) and streams it back to the client in real-time via Server-Sent Events (SSE).
3. **Execution & State Persistence:** Users write and validate their code (executed via the E2B sandbox). Session logs, attempt scores, and subtopic mastery are recorded by the `state` module and persisted locally to `progress.json` and the `sessions/` directory.

## Environment & Setup
**Prerequisites:** Python 3.x, pip

**Environment Variables:**
Create a `.env` file in the root directory:
```env
NVIDIA_API_KEY=your_key_here
MODEL=moonshotai/kimi-k2-instruct-0905
```

**Start Commands:**
```bash
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```
Access the application at `http://localhost:8000`.

## Development Conventions
* **Educational Constraints:** The AI must prioritize reasoning over results and is strictly prompted to never output full code solutions.
* **Streaming Responses:** Use Server-Sent Events (SSE) wrapped around token generators to deliver low-latency AI responses.
* **State Management:** Simple, local JSON-based persistence (no DB). 

## Known Issues / Debt
* **Single-User Architecture:** There is currently no multi-user authentication system.
* **Data Persistence:** Relies solely on JSON file storage (`progress.json`), which lacks concurrency controls and will not scale for multiple users.
* **External Dependencies:** The platform's responsiveness heavily depends on the latency of the external Nvidia LLM inference API and E2B sandbox.
* **Sandbox Implementation Incomplete:** While E2B is mentioned in the documentation, the `sandbox/` directory and E2B SDK are currently absent from the core project files and `requirements.txt`.
