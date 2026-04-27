# Crack DSA - Coaching Platform

DSA Coach is a web application designed to help software engineers prepare for FAANG interviews. It provides intelligent, focused, and interactive coaching driven by heavily structured System Prompts.

## System Architecture & Technical Details

The platform pairs a FastAPI web server with streaming capabilities to deliver real-time feedback using an advanced reasoning LLM (`moonshotai/kimi-k2-instruct-0905`). The LLM is configured via `langchain_nvidia_ai_endpoints` with specific instructions enabling "thinking" and "reasoning_effort: high" to ensure the generation of robust and intelligent coaching feedback.

### Prompt Engineering & Coaching Modes

The system operates across several specialized interaction modes, each powered by a distinct system prompt ensuring the AI acts strictly as a coach rather than a solution provider. These prompts heavily emphasize conceptual understanding over raw code generation.

#### 1. Teaching Prompt (`TEACH_SYSTEM`)
**Objective**: Introduce and teach new DSA concepts.
- **Constraints**: NEVER write executable code. Explanations must rely exclusively on plain English, pseudocode, and ASCII diagrams.
- **Pedagogy**: Instructed to begin with intuition and real-world analogies before diving into the technical aspects (time/space complexity, common patterns, pitfalls).
- **Assessment**: Terminates every teaching session with 2-3 targeted self-check questions to validate user understanding.

#### 2. Q&A Prompt (`TEACH_QA_SYSTEM`)
**Objective**: Handle follow-up questions during learning.
- **Focus**: Keeps the user on-topic regarding the specific data structure or algorithm being discussed.
- **Constraints**: Operates under the strict no-executable-code constraint, only delivering pseudocode and diagrams. Includes instructions to gracefully redirect off-topic inquiries.

#### 3. Evaluation Prompt (`EVAL_SYSTEM`)
**Objective**: Act as a strict FAANG interviewer evaluating candidate approaches.
- **Inputs**: Problem title, difficulty, and the user's plain English algorithmic approach.
- **Output Format**: Strictly enforces a structured response block parsing the user's approach:
  - `VERDICT`: `CORRECT | PARTIALLY CORRECT | WRONG`
  - `SCORE`: `0-100`
  - `STRENGTHS`: Bulleted list
  - `GAPS`: Bulleted list
  - `FOLLOWUP`: One targeted follow-up question.
- **Scoring Logic**: Grants partial credit for correct intuitions paired with suboptimal time/space complexities. Instructed to guide via questioning rather than revealing the final solution.

#### 4. Progressive Hint Prompt (`HINT_SYSTEM`)
**Objective**: Deliver layered hints based on user requests without spoiling the answer.
- **Level 1**: Gentle nudges towards the correct data structure or pattern.
- **Level 2**: High-level algorithmic approach without minor details.
- **Level 3**: Step-by-step walkthrough generation using pseudocode.

#### 5. Deep-Dive Follow-up Prompt (`FOLLOWUP_SYSTEM`)
**Objective**: Stress-test the user's algorithmic intuition on completed or reviewed problems.
- **Inputs**: The problem title, the user's previous approach, and the prior evaluation feedback.
- **Probing Rules**: Targets missed edge cases, evaluates approach scaling (e.g., 10x data volumes), and discusses alternative approaches along with their comparative trade-offs. Responses are constrained to 2-3 sentences.

#### 6. Targeted Review Prompt (`WEAK_AREA_SYSTEM`)
**Objective**: Remediate topics where the user has scored below a 60% proficiency threshold.
- **Structure**: Initiates a 5-minute targeted session using fresh real-world analogies (different from what was previously taught). Automatically provides a conceptual warm-up problem (without the solution) to challenge the student's pattern recognition.

---

## Codebase API & Methods

### `llm.py`
**get_client**
What it does: Creates and returns a `ChatNVIDIA` model instance configured for the system.
Argument: None
Output: `ChatNVIDIA` configured client.
Example:
```python
client = get_client()
```
Output: `<ChatNVIDIA model="moonshotai/kimi-k2-instruct-0905">`

**stream_tokens**
What it does: Formats message history and yields an asynchronous stream of response chunks from the LLM.
Argument: `client: ChatNVIDIA`, `system: str`, `history: list[dict]`
Output: `Generator[str, None, None]`
Example:
```python
for token in stream_tokens(client, "You are a bot", []): 
    print(token)
```
Output: `"Hello", " there!"`

### `state/progress.py`
**load**
What it does: Loads the user state and progress dictionary from `progress.json`.
Argument: None
Output: `dict`
Example:
```python
progress = load()
```
Output: `{"subtopics_learned": {}, "problems": {}, ...}`

**save**
What it does: Saves the current state dictionary dynamically back into `progress.json`.
Argument: `data: dict`
Output: `None`
Example:
```python
save(progress)
```
Output: (Updates json file successfully)

**get_weak_topics**
What it does: Gets topics where user performance falls under a specified threshold.
Argument: `progress: dict`, `threshold: int = 60`
Output: `list[str]`
Example:
```python
weak = get_weak_topics(progress, 60)
```
Output: `["arrays_101"]`

**update_score**
What it does: Updates the subtopic score computing exponentially weighted moving average dynamically.
Argument: `progress: dict`, `topic_id: str`, `new_score: int`
Output: `None`
Example:
```python
update_score(progress, "arrays_101", 80)
```
Output: `None`

**record_problem_attempt**
What it does: Records internal attempts metrics, updates optimal high-scores automatically, and registers chronological timestamps.
Argument: `progress: dict`, `problem_id: str`, `score: int`
Output: `None`
Example:
```python
record_problem_attempt(progress, "two_sum", 100)
```
Output: `None`

### `server.py`
**root**
What it does: Static entry point resolving requests directly to the user-agent interface web app (`app.html`).
Argument: None
Output: `HTMLResponse`
Example:
```python
await root()
```
Output: `<HTMLResponse ...>`

**get_curriculum**
What it does: Endpoint returning structured JSON nodes containing problem and topic modules.
Argument: None
Output: `JSONResponse | dict`
Example:
```python
await get_curriculum()
```
Output: `{"chapters": [...]}`

**get_progress**
What it does: Endpoint wrapping the internal `load()` function to distribute user state safely across front end hooks.
Argument: None
Output: `dict`
Example:
```python
await get_progress()
```
Output: `{"subtopics_learned": {}, ...}`

**mark_learned**
What it does: Endpoint storing explicitly marked topic segment completion attributes.
Argument: `req: MarkLearnedRequest`
Output: `dict`
Example:
```python
await mark_learned(MarkLearnedRequest(subtopic_id="arrays_101"))
```
Output: `{"ok": True}`

**record_attempt**
What it does: Evaluates algorithmic submissions by calculating progress differences and logging output responses asynchronously to logs.
Argument: `req: RecordAttemptRequest`
Output: `dict`
Example:
```python
await record_attempt(RecordAttemptRequest(problem_id="1", subtopic_id="1", score=100))
```
Output: `{"ok": True}`

**sse**
What it does: Converts a raw String token generator array format incrementally resolving partial frames over `text/event-stream`.
Argument: `gen: Generator`
Output: `StreamingResponse`
Example:
```python
response = sse(stream_tokens(...))
```
Output: `<StreamingResponse />`

**learn_start** , **learn_qa** , **evaluate** , **hint** , **review_start**
What it does: Generates context, manages prompt parameters, and forwards request payload streams efficiently resolving API streaming buffers simultaneously referencing dynamically fetched static `prompts_text` prompts.
Argument: Valid target schemas (`LearnRequest`, `QARequest`, `EvalRequest`, etc).
Output: `StreamingResponse`
Example:
```python
await learn_start(LearnRequest(...))
```
Output: `<StreamingResponse ...>`

**Pydantic Schema Models** (`LearnRequest`, `QARequest`, `EvalRequest`, `HintRequest`, `ReviewRequest`, `MarkLearnedRequest`, `RecordAttemptRequest`)
What it does: Represents structured validations bounding inbound endpoints parsing safety payload boundaries accurately enforcing strict mapping invariants consistently.
Argument: Payload attributes logic dictionaries.
Output: Initialized Pydantic Class instance validation objects safely tracking payloads.
Example:
```python
HintRequest(problem_title="twosum", hint_level=1)
```
Output: `HintRequest(problem_title='twosum', hint_level=1)`

---

## User Interface & Frontend Workflow

The web interface (`static/app.html`) is a standalone plain HTML/CSS/JS Single Page Application (SPA) structured around a robust view-swapping logic mapped via a central sidebar navigation pipeline. The JavaScript utilizes modular layout functions binding the API layer utilizing Server-Sent Events (SSE).

### 1. The Core Infrastructure (`init()` & `streamSSE()`)
The app initializes by fetching JSON configuration properties from `/api/curriculum` and `/api/progress` mapping it into global array maps closures (`curriculum`, `progress`). For LLM streaming integration, a custom async method resolving nested `text/event-stream` iterations triggers payload buffers manually executing logic bounded over newline delineation arrays (`\\n\\n`) allowing for synchronized live typewriter chat rendering in the DOM bypassing string concatenation latency.

### 2. Dashboard View (`view-dashboard`)
Provides an aggregate high-level overview resolving score arrays mathematically. Progress attributes calculate cumulative parameters binding dynamic color thresholds algorithmically (e.g., `<= 60 = Red/Weak Area` or `>= 80 = Green/Learned`).

### 3. Learn View (`view-learn`)
Organizes curriculum branches dynamically referencing arrays iteratively over interactive DOM components triggering async sessions initializing `startLearn()`. The active chat container streams LLM conversational rules safely. Hard-coded shortcut "Quick Action" parameter strings explicitly pipe injection instructions (Analogy, Pitfalls, Time/Space calculations) parsing logic streams correctly routing dynamic topic history endpoints.

### 4. Practice View (`view-practice`)
Isolates candidate inputs validating structural evaluation patterns dynamically tracking completion conditions natively. The candidate types logical plain text in `practice-input-section`. The evaluation execution binds `/api/practice/evaluate` output logic formatting dynamically, actively executing regex (`SCORE:\\s*(\\d+)`) parsing routines converting the scalar evaluation bounds logically mapping circular SVG SVG gauge offsets asynchronously rendering dynamically based on algorithm accuracy constraints calculation paths. Supports rendering dynamic recursive hint streams.

### 5. Review View (`view-review`)
Dynamic routing filters targeting explicit algorithmic properties calculating weak area thresholds directly mapping under (`score < 60`). The review UI dispatches endpoint contexts to target weak structures, routing prompt contexts asynchronously parsing array topics formatting session inputs smoothly overriding session states safely.