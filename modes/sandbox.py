"""
sandbox.py — Standalone E2B sandboxed code execution tool for LLM integration.

This module is SELF-CONTAINED. It does NOT import or modify anything from the
rest of the DSA Coach codebase (server.py, llm.py, state/, modes/, etc.).

Usage (standalone test):
    python sandbox.py

Usage (as a tool from any other module — import only this):
    from sandbox import execute_python, TOOL_DEFINITION, run_tool_loop

Requirements:
    pip install e2b-code-interpreter langchain-nvidia-ai-endpoints python-dotenv

Environment variables required (add to your .env):
    E2B_API_KEY=your_e2b_key_here
    NVIDIA_API_KEY=your_nvidia_key_here   # already present in the project .env
"""

import os
import json
from typing import Any
from dotenv import load_dotenv

load_dotenv()

# ── E2B import guard ─────────────────────────────────────────────────

try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    raise ImportError(
        "e2b-code-interpreter is not installed.\n"
        "Run: pip install e2b-code-interpreter"
    )

# ── LLM import (mirrors llm.py style but is fully independent) ───────

try:
    from langchain_nvidia_ai_endpoints import ChatNVIDIA
except ImportError:
    raise ImportError(
        "langchain-nvidia-ai-endpoints is not installed.\n"
        "Run: pip install langchain-nvidia-ai-endpoints"
    )

# ── Config ────────────────────────────────────────────────────────────

E2B_API_KEY   = os.environ.get("E2B_API_KEY", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL_NAME     = "moonshotai/kimi-k2-instruct-0905"   # same model as the project

# ── Tool definition (OpenAI-compatible function-calling schema) ───────

TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute Python code in a secure, isolated E2B cloud sandbox "
            "and return stdout, stderr, and any rich cell outputs. "
            "Use this to run, test, or verify algorithmic solutions, "
            "compute complexity experiments, trace data structure operations, "
            "or validate any code snippet."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Valid Python code to execute. "
                        "The sandbox persists across calls in the same session, "
                        "so variables defined in earlier cells are available here."
                    ),
                },
                "install_packages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of pip packages to install before running the code, "
                        "e.g. ['numpy', 'matplotlib']. Leave empty if not needed."
                    ),
                    "default": [],
                },
            },
            "required": ["code"],
        },
    },
}

# Langchain-compatible tool spec (for use with .bind_tools())
LANGCHAIN_TOOL_SPEC: dict[str, Any] = {
    "name": TOOL_DEFINITION["function"]["name"],
    "description": TOOL_DEFINITION["function"]["description"],
    "parameters": TOOL_DEFINITION["function"]["parameters"],
}

# ── Core execution function ───────────────────────────────────────────

def execute_python(
    code: str,
    install_packages: list[str] | None = None,
    sandbox: "Sandbox | None" = None,
) -> dict[str, Any]:
    """
    Execute Python code in an E2B sandbox.

    Args:
        code:             The Python code to run.
        install_packages: Optional pip packages to install first.
        sandbox:          An existing Sandbox instance to reuse (for multi-turn
                          sessions). If None, a fresh sandbox is created and
                          closed after execution.

    Returns:
        A dict with keys:
            stdout    — captured print() output (str)
            stderr    — error output / tracebacks (str)
            result    — final cell expression value as text (str | None)
            error     — structured error info if execution failed (dict | None)
            success   — True if no exceptions were raised (bool)
    """
    if not E2B_API_KEY:
        return {
            "stdout": "",
            "stderr": "",
            "result": None,
            "error": {"name": "ConfigError", "value": "E2B_API_KEY is not set in environment."},
            "success": False,
        }

    owns_sandbox = sandbox is None

    try:
        if owns_sandbox:
            sandbox = Sandbox(api_key=E2B_API_KEY)

        # Install any requested packages first
        if install_packages:
            pkg_str = " ".join(install_packages)
            install_exec = sandbox.run_code(f"import subprocess; subprocess.run(['pip', 'install', '{pkg_str}', '-q'], check=True)")
            if install_exec.error:
                return {
                    "stdout": "",
                    "stderr": f"Package install failed: {install_exec.error.value}",
                    "result": None,
                    "error": {"name": install_exec.error.name, "value": install_exec.error.value},
                    "success": False,
                }

        execution = sandbox.run_code(code)

        stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else ""
        stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
        result = execution.text  # final expression value (like a Jupyter cell)

        if execution.error:
            return {
                "stdout": stdout,
                "stderr": stderr,
                "result": result,
                "error": {
                    "name": execution.error.name,
                    "value": execution.error.value,
                    "traceback": execution.error.traceback,
                },
                "success": False,
            }

        return {
            "stdout": stdout,
            "stderr": stderr,
            "result": result,
            "error": None,
            "success": True,
        }

    except Exception as exc:
        return {
            "stdout": "",
            "stderr": str(exc),
            "result": None,
            "error": {"name": type(exc).__name__, "value": str(exc)},
            "success": False,
        }
    finally:
        if owns_sandbox and sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                pass


# ── Tool dispatcher (maps LLM tool call → execute_python) ─────────────

def dispatch_tool_call(
    tool_name: str,
    tool_args: dict[str, Any],
    sandbox: "Sandbox | None" = None,
) -> str:
    """
    Route an LLM-generated tool call to the correct handler and return
    a JSON string to feed back into the message history as a tool result.
    """
    if tool_name == "execute_python":
        result = execute_python(
            code=tool_args.get("code", ""),
            install_packages=tool_args.get("install_packages", []),
            sandbox=sandbox,
        )
        return json.dumps(result)
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ── Session class for multi-turn use ─────────────────────────────────

class SandboxSession:
    """
    A persistent sandbox session.

    Keeps the same E2B sandbox alive across multiple LLM tool calls so that
    variables, imports, and state carry over between turns — exactly like a
    Jupyter notebook session.

    Usage:
        with SandboxSession() as session:
            result = session.run("x = 42")
            result = session.run("print(x * 2)")   # x is still in scope
    """

    def __init__(self):
        if not E2B_API_KEY:
            raise EnvironmentError("E2B_API_KEY is not set.")
        self._sandbox = Sandbox(api_key=E2B_API_KEY)

    def run(self, code: str, install_packages: list[str] | None = None) -> dict[str, Any]:
        return execute_python(code, install_packages=install_packages, sandbox=self._sandbox)

    def close(self):
        try:
            self._sandbox.kill()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Agentic tool-loop ─────────────────────────────────────────────────

SANDBOX_SYSTEM_PROMPT = """You are a DSA coaching assistant with access to a Python code execution sandbox.

When you need to:
- Verify the correctness of an algorithm
- Measure runtime or memory empirically
- Trace through data structure operations step by step
- Demonstrate time complexity with concrete benchmarks

...you MUST use the execute_python tool to run real code rather than guessing.

Rules:
- Never fabricate execution output. Always run code to verify.
- After execution, explain the output clearly in plain English.
- If code fails, diagnose the error and retry with a fix.
- Keep code cells focused — one concept per cell.
- Do NOT use execute_python for explanations or pseudocode — only for runnable Python.
"""


def run_tool_loop(
    user_message: str,
    max_iterations: int = 6,
    verbose: bool = True,
) -> str:
    """
    Run a full agentic tool-use loop:
      user → LLM → [tool call → execute_python → tool result → LLM] → final answer

    Args:
        user_message:   The user's question or task.
        max_iterations: Max LLM→tool→LLM cycles before forcing a stop.
        verbose:        Print intermediate steps to stdout.

    Returns:
        The LLM's final text response.
    """
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY is not set."

    llm = ChatNVIDIA(
        model=MODEL_NAME,
        api_key=NVIDIA_API_KEY,
        temperature=0,
    ).bind_tools([LANGCHAIN_TOOL_SPEC])

    messages: list[dict[str, Any]] = [
        {"role": "system",  "content": SANDBOX_SYSTEM_PROMPT},
        {"role": "user",    "content": user_message},
    ]

    with SandboxSession() as session:
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n[sandbox.py] LLM call #{iteration + 1}...")

            response = llm.invoke(messages)

            # Append assistant message
            messages.append({"role": "assistant", "content": response.content or "", "tool_calls": getattr(response, "tool_calls", [])})

            tool_calls = getattr(response, "tool_calls", []) or []

            # No tool calls → final answer
            if not tool_calls:
                if verbose:
                    print("[sandbox.py] No tool calls — returning final answer.")
                return response.content or ""

            # Process each tool call
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_id   = tc.get("id", "call_0")

                if verbose:
                    print(f"[sandbox.py] Tool call: {tool_name}({json.dumps(tool_args)[:120]}...)")

                tool_result_str = dispatch_tool_call(tool_name, tool_args, sandbox=session._sandbox)
                tool_result     = json.loads(tool_result_str)

                if verbose:
                    if tool_result["success"]:
                        print(f"[sandbox.py] ✓ stdout: {tool_result['stdout'][:200]}")
                    else:
                        print(f"[sandbox.py] ✗ error: {tool_result.get('error')}")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_id,
                    "name":         tool_name,
                    "content":      tool_result_str,
                })

        return "[sandbox.py] Max iterations reached without a final answer."


# ── Standalone test ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DSA Coach — E2B Sandbox Tool (standalone test)")
    print("=" * 60)

    # Test 1: raw execute_python (no LLM)
    print("\n[Test 1] Direct sandbox execution — Two Sum O(n) solution")
    with SandboxSession() as s:
        out = s.run("""
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        complement = target - n
        if complement in seen:
            return [seen[complement], i]
        seen[n] = i
    return []

# Test cases
assert two_sum([2, 7, 11, 15], 9) == [0, 1]
assert two_sum([3, 2, 4], 6)      == [1, 2]
assert two_sum([3, 3], 6)         == [0, 1]
print("All test cases passed!")
print(f"Result: {two_sum([2, 7, 11, 15], 9)}")
""")
        print("stdout:", out["stdout"])
        print("success:", out["success"])
        if not out["success"]:
            print("error:", out["error"])

    # Test 2: full agentic loop (requires both API keys)
    if E2B_API_KEY and NVIDIA_API_KEY:
        print("\n[Test 2] Agentic tool loop — complexity comparison")
        answer = run_tool_loop(
            "Write Python code to empirically compare the runtime of O(n²) bubble sort "
            "vs O(n log n) merge sort on a list of 1000 random integers, then print which is faster.",
            verbose=True,
        )
        print("\n[Final LLM Answer]\n", answer)
    else:
        print("\n[Test 2] Skipped — set E2B_API_KEY and NVIDIA_API_KEY to test the full loop.")

    print("\n" + "=" * 60)
    print("sandbox.py test complete.")